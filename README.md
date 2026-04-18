# Pocket-Agent Submission

Offline tool-calling assistant built for the hackathon constraints.

## Setup Instructions

1. Install dependencies

   python -m pip install -r requirements.txt

2. Generate training data (curated + synthetic)

   python data/generate_data.py --out data/train.jsonl --n 2000 --manual data/manual_examples.jsonl

3. Optional hard-gate overlap check if public test is available

   python data/check_overlap.py --train data/train.jsonl --public starter/public_test.jsonl

4. Train LoRA adapter

   python train.py --model_id Qwen/Qwen2.5-0.5B-Instruct --data data/train.jsonl --out models/adapter

5. Quantize

   Bonus attempt (<=250 MB):
   python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q3_k_s --max_mb 250

   Gate-safe fallback (<=500 MB):
   python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q4_k_m

6. Run demo

   MODEL_PATH=models/quantized/model.gguf python demo/app.py

7. Grader contract check

   from inference import run
   print(run("weather in Lahore in C", []))

## Design Decisions

- Tool output is always constrained to either exact <tool_call>{...}</tool_call> or plain refusal text.
- Inference has lightweight refusal guards for ambiguous no-context prompts to avoid -0.5 penalties.
- Model output is canonicalized and validated for schema, JSON shape, and key argument formats.
- Training data mixes deterministic curated examples with synthetic variation to cover all grading slices.

## Model Choices

- Primary model: Qwen2.5-0.5B-Instruct
- Reason 1: safely below <=2B hard gate
- Reason 2: strong instruction-following and structured output behavior for its size
- Reason 3: practical quantization path to <=500MB and realistic <=250MB attempts

## Slice Coverage Strategy

- Slice A (in-distribution): direct tool patterns in synthetic generator
- Slice B (paraphrased): lexical paraphrases in synthetic templates and curated prompts
- Slice C (adversarial): typos, code-switch (Hindi/Urdu/Spanish/Arabic mixed), ambiguity, hallucination-bait entities
- Slice D (refusals + multi-turn): impossible tools, ambiguous references without history, and 2-3 turn carry-over examples

## Hard Gates Checklist

- Adapter loads on declared <=2B base model in transformers: yes (train.py + quantize.py path)
- Quantized model <=500MB: q4_k_m path
- Mean latency <=200ms target: low-token deterministic inference settings and small quantized model
- Zero prompt overlap with public test: check_overlap.py script + --avoid generation option
- No network imports in inference.py: yes
- Demo launches and accepts input: demo/app.py Gradio ChatInterface

## Bonus Point Plan

- <=250MB bonus: first quantize with q3_k_s and --max_mb 250; fallback to q4_k_m if needed
- README debugging insight bonus: see Error Analysis section below
- Slice C performance bonus strategy: heavy curated adversarial set in data/manual_examples.jsonl plus canonical argument normalization

## What Worked

- Curated adversarial examples significantly improved refusal and multi-turn consistency.
- Canonicalizing tool payloads reduced malformed JSON outputs at inference time.
- Early refusal for ambiguous no-history references prevented accidental negative scoring.
- q3_k_s often meets size target while still preserving usable tool-call behavior.

## What Did Not Work

- Pure synthetic random data without curated seeds underperformed on code-switched edge prompts.
- Raw model output without post-validation produced occasional malformed tool payloads.
- Aggressive size-only quantization can hurt argument fidelity on adversarial prompts.

## Error Analysis (Specific Debugging Insight)

Issue observed:
The model occasionally emitted a tool call for prompts like "convert that to euros" when there was no prior context.

Why this was costly:
Those cases should be refusal decisions; a wrong tool call can incur negative scoring.

Debugging insight and fix:
- Added explicit ambiguous-reference detection in inference.py for tokens like that/it/same/there when no prior tool context exists.
- Added strict tool-call JSON extraction and schema canonicalization so malformed outputs are rejected into refusal text.

Resulting behavior change:
- Ambiguous no-history prompts now deterministically return refusal text.
- Tool outputs are normalized, reducing malformed JSON risk and preserving argument fidelity.

## GitHub and Colab Workflow for Submission

Submission is by GitHub repository link, not by uploading zip.

Recommended flow:
1. Push code to GitHub.
2. In Colab, clone repo, train, quantize, smoke test.
3. Commit adapter artifacts from Colab only if needed.
4. Push final commit; submit repository URL on platform.

Push from local:

   git init
   git add .
   git commit -m "hackathon submission"
   git branch -M main
   git remote add origin https://github.com/ahmedbilal008/vyro-hackathon-submission.git
   git push -u origin main

Push from Colab after training (optional):

   !git config user.email "you@example.com"
   !git config user.name "Your Name"
   !git add models/adapter
   !git commit -m "add trained adapter"
   !git push

Note:
- Quantized files are usually not required in GitHub if quantize.py reproduces them.
- Keep quantized model local in Colab for evaluation/demo, and commit only if size and policy allow.
