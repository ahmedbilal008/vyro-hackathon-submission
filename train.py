import argparse
import json
import os
from typing import List, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. "
    "For valid tool requests, output exactly one tool call as: "
    "<tool_call>{\"tool\":...,\"args\":...}</tool_call>. "
    "For refusals, output plain text with no tool call."
)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_chatml(messages: List[Dict], add_generation_prompt: bool) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_features(example, tokenizer, max_length: int):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + example["messages"]
    prompt_text = format_chatml(messages, add_generation_prompt=True)
    full_messages = messages + [{"role": "assistant", "content": example["answer"]}]
    full_text = format_chatml(full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)

    input_ids = full["input_ids"][:max_length]
    attention_mask = full["attention_mask"][:max_length]

    labels = [-100] * len(prompt_ids) + full["input_ids"][len(prompt_ids):]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_batch(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--out", default="models/adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_jsonl(args.data)
    dataset = Dataset.from_list(raw)
    original_cols = dataset.column_names
    dataset = dataset.map(
        lambda ex: build_features(ex, tokenizer, args.max_length),
        remove_columns=original_cols,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    def collate(batch):
        return collate_batch(batch, tokenizer.pad_token_id)

    args_train = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(model=model, args=args_train, train_dataset=dataset, data_collator=collate)
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}")


if __name__ == "__main__":
    main()
