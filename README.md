# Pocket-Agent Submission

Fine-tuned offline tool-calling assistant for the hackathon grader.

## Setup Instructions

Run from repository root:

```bash
python -m pip install -r requirements.txt
python data/generate_data.py --out data/train.jsonl --n 2000 --manual data/manual_examples.jsonl
python train.py --model_id Qwen/Qwen2.5-0.5B-Instruct --data data/train.jsonl --out models/adapter
python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q3_k_s --max_mb 250
```

If the q3 run does not pass size/quality needs, use fallback:

```bash
python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q4_k_m
```

Smoke test and demo:

```bash
python -c "from inference import run; print(run('weather in Lahore in C', []))"
MODEL_PATH=models/quantized/model.gguf python demo/app.py
```

## Design Decisions

- Enforced exact output mode: valid <tool_call>{...}</tool_call> or plain refusal text.
- Added inference-side schema validation/canonicalization to reduce malformed JSON and wrong-arg scoring drops.
- Added refusal guard for ambiguous no-context prompts to avoid negative scoring.
- Used Qwen2.5-0.5B-Instruct with q3 first (for <=250MB attempt) and q4 fallback (for <=500MB safety).
- Mixed synthetic + curated manual edge cases to cover in-distribution, paraphrase, adversarial, and multi-turn/refusal slices.

## Model Choice

- Base model: Qwen2.5-0.5B-Instruct
- Why: <=2B gate safe, strong structured-output behavior for size, quantizes well for Colab CPU constraints.

## What Worked

- Curated adversarial examples improved code-switched and ambiguous prompt handling.
- JSON canonicalization improved exact-format and argument consistency.
- Refusal guard reduced false tool calls on no-context references.

## What Did Not Work

- Purely random synthetic data was weak on adversarial prompts.
- No post-validation caused malformed outputs in some generations.
- Over-aggressive quantization can reduce argument fidelity.

## Colab Flow (Exact Steps)

1. Open Colab T4 runtime.
2. Run these commands in order:

```python
!git clone https://github.com/ahmedbilal008/vyro-hackathon-submission.git
%cd vyro-hackathon-submission
!pip -q install -r requirements.txt
!python data/generate_data.py --out data/train.jsonl --n 2000 --manual data/manual_examples.jsonl
!python train.py --model_id Qwen/Qwen2.5-0.5B-Instruct --data data/train.jsonl --out models/adapter
!python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q3_k_s --max_mb 250
```

3. If the last command fails or accuracy is not stable, run fallback:

```python
!python quantize.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter models/adapter --out models/quantized/model.gguf --quant q4_k_m
```

4. Check and run demo:

```python
from inference import run
print(run("weather in Lahore in C", []))
import os
os.environ["MODEL_PATH"] = "models/quantized/model.gguf"
!python demo/app.py
```

## Colab Commit Steps (What You Should Push)

Push code + trained adapter. Quantized file can stay uncommitted since grader quantizes from adapter.

```python
!git config user.name "Ahmed Bilal"
!git config user.email "your-email@example.com"
!git add models/adapter
!git commit -m "add trained adapter"
```

Then push with a token:

```python
import getpass
token = getpass.getpass("GitHub token: ")
!git remote set-url origin https://{token}@github.com/ahmedbilal008/vyro-hackathon-submission.git
!git push origin main
```

## Submission

Submit this GitHub repository link on the hackathon platform:

https://github.com/ahmedbilal008/vyro-hackathon-submission
