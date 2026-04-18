# Pocket-Agent Submission

## Quick Run

Use [colab.ipynb](colab.ipynb) as the single entry point.

1. Open [colab.ipynb](colab.ipynb) in Colab.
2. Run all cells top to bottom.
3. Wait for the notebook readiness checks to print PASS.
4. Submit this repository link on the platform:

https://github.com/ahmedbilal008/vyro-hackathon-submission

Notes:

- The notebook already handles setup, data generation, training, quantization, public eval, and demo launch.
- If quantization is slow, keep the cell running unless it shows no new logs for a long time.

## Design Choices

- Base model is kept <=2B (Qwen2.5-0.5B-Instruct) to satisfy hard gate constraints.
- Output is constrained to exact `<tool_call>{...}</tool_call>` or plain refusal text.
- Inference applies schema validation and canonicalization to reduce malformed outputs.
- Data combines synthetic templates and curated adversarial/manual examples for all grading slices.
- Quantization prioritizes a stable gate-safe path (`q4_k_m`) with a fallback (`q4_0`).

## Model Choice

- Base model: Qwen2.5-0.5B-Instruct
- Quantization: q4_k_m primary, q4_0 fallback
- Inference runtime target: Colab CPU with GGUF

## What Worked

- Curated adversarial prompts improved robustness on typos and code-switching.
- Refusal guard logic improved ambiguous no-history behavior.
- Canonical JSON formatting improved tool-call consistency.

## What Didn't

- Purely random synthetic data was weak on adversarial edge cases.
- Over-aggressive compression can reduce argument fidelity.
