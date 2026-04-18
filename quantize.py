import argparse
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_llama_cpp(path):
    if not os.path.exists(path):
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp", path])
    quantize_bin = os.path.join(path, "llama-quantize")
    if not os.path.exists(quantize_bin):
        run(["make", "-C", path, "llama-quantize"])
    if not os.path.exists(quantize_bin):
        legacy_bin = os.path.join(path, "quantize")
        if not os.path.exists(legacy_bin):
            run(["make", "-C", path, "quantize"])


def merge_lora(base_id, adapter_dir, merged_dir):
    os.makedirs(merged_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)


def convert_and_quantize(merged_dir, out_path, llama_dir, quant_type):
    f16_path = out_path + ".f16.gguf"
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_dir, "quantize")
    run([sys.executable, convert_script, "--outtype", "f16", "--outfile", f16_path, merged_dir])
    run([quantize_bin, f16_path, out_path, quant_type])


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="models/adapter")
    parser.add_argument("--merged", default="models/merged")
    parser.add_argument("--out", default="models/quantized/model.gguf")
    parser.add_argument("--llama_dir", default="llama.cpp")
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--max_mb", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ensure_llama_cpp(args.llama_dir)
    merge_lora(args.base, args.adapter, args.merged)
    convert_and_quantize(args.merged, args.out, args.llama_dir, args.quant)
    size_mb = file_size_mb(args.out)
    print(f"Wrote quantized model to {args.out} ({size_mb:.2f} MB)")
    if args.max_mb > 0 and size_mb > args.max_mb:
        raise SystemExit(f"Quantized model exceeds size limit: {size_mb:.2f} MB > {args.max_mb:.2f} MB")


if __name__ == "__main__":
    main()
