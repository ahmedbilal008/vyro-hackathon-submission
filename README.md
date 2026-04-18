# Pocket-Agent Submission

This repository is prepared for the hackathon grading flow on clean Colab T4:
clone repo -> load adapter on base -> quantize -> score -> demo launch.

## Setup Instructions (Colab-First)

Run in this exact order in Colab:

```python
!git clone https://github.com/ahmedbilal008/vyro-hackathon-submission.git
%cd vyro-hackathon-submission
!pip -q install --upgrade pip setuptools wheel
!pip -q install --prefer-binary -r requirements.txt

!python starter/build_starter_files.py --manual data/manual_examples.jsonl --out starter

!python data/generate_data.py --out data/train.jsonl --n 2000 --manual data/manual_examples.jsonl
!python train.py --model_id Qwen/Qwen2.5-0.5B-Instruct --data data/train.jsonl --out models/adapter

# Fast gate-safe quantization (<=500MB)
!python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q4_k_m
```

If install appears stuck on llama-cpp-python for more than 8-10 minutes, interrupt and run:

```python
!pip -q install --upgrade pip setuptools wheel
!pip -q install --prefer-binary "llama-cpp-python==0.2.90"
!pip -q install --prefer-binary -r requirements.txt
```

If q4_k_m fails, use this fallback:

```python
!python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q4_0
```

Smoke test + demo:

```python
from inference import run
print(run("weather in Lahore in C", []))

!python eval_public.py --test starter/public_test.jsonl --out eval_public_summary.json

import os
os.environ["MODEL_PATH"] = "models/quantized/model.gguf"
!python demo/app.py
```

If you want to download adapter from Colab instead of pushing from Colab git:

```python
!zip -r adapter_only.zip models/adapter
from google.colab import files
files.download("adapter_only.zip")
```

## Design Decisions

- Base model is Qwen2.5-0.5B-Instruct to stay safely under <=2B.
- Output is constrained to exact <tool_call>{...}</tool_call> or plain refusal.
- Inference validates/canonicalizes tool JSON to reduce malformed output and arg errors.
- Ambiguous no-history prompts are refused early to avoid negative scoring.
- Data mixes synthetic and curated adversarial/manual examples for all grading slices.
- Quantization strategy is q4_k_m first for fastest stable run, with q4_0 fallback.

## Model Choices

- Base model: Qwen2.5-0.5B-Instruct
- Quantization: q4_k_m primary, q4_0 fallback
- Runtime target: Colab CPU inference via GGUF + llama.cpp backend

## What Worked

- Curated adversarial examples improved Slice C behavior (typos/code-switch/ambiguity).
- Canonical tool JSON reduced malformed outputs.
- Refusal guard improved Slice D refusal correctness.

## What Didn't

- Fully random synthetic data was weak on adversarial prompts.
- No output post-validation led to occasional malformed tool-call outputs.
- Aggressive quantization can reduce argument fidelity.

## Colab Commit Steps (After Training)

Commit adapter from Colab, then push:

```python
!git config user.name "Ahmed Bilal"
!git config user.email "your-email@example.com"
!git add models/adapter
!git commit -m "add trained adapter"

import getpass
token = getpass.getpass("GitHub token: ")
!git remote set-url origin https://{token}@github.com/ahmedbilal008/vyro-hackathon-submission.git
!git push origin main
```

Quantized file is optional to push; grader can regenerate it from adapter via quantize.py.

## Starter/Eval Files Included

- starter/public_test.jsonl (40 dev examples)
- starter/teacher_examples.jsonl (20 seed examples)
- starter/tool_schemas.json (final 5-tool schema)
- starter/eval_harness_contract.py (local grader-style scorer)
- eval_public.py (one-command public eval runner)

## Submission

Submit this public repository link:

https://github.com/ahmedbilal008/vyro-hackathon-submission
