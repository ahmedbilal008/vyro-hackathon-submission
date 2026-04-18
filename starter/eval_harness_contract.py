import argparse
import json
import math
import re
import time
from collections import defaultdict


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)


def parse_tool_call(text):
    if not isinstance(text, str):
        return None
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if "tool" not in payload or "args" not in payload:
        return None
    return payload


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_close(pred, gold):
    if not (is_number(pred) and is_number(gold)):
        return False
    if gold == 0:
        return abs(pred - gold) <= 0.01
    return abs(pred - gold) <= abs(gold) * 0.01


def args_exact(pred_args, gold_args):
    if not isinstance(pred_args, dict) or not isinstance(gold_args, dict):
        return False
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key, gold_val in gold_args.items():
        pred_val = pred_args.get(key)
        if is_number(gold_val):
            if not number_close(pred_val, gold_val):
                return False
        else:
            if pred_val != gold_val:
                return False
    return True


def derive_prompt_history(example):
    if "prompt" in example:
        return example.get("prompt", ""), example.get("history", [])
    msgs = example.get("messages", [])
    if not msgs:
        return "", []
    prompt = msgs[-1].get("content", "")
    history = msgs[:-1]
    return prompt, history


def score_example(run_fn, example):
    prompt, history = derive_prompt_history(example)
    expected = example.get("answer", "")

    t0 = time.perf_counter()
    pred = run_fn(prompt, history)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    gold_tool = parse_tool_call(expected)
    pred_tool = parse_tool_call(pred)

    if gold_tool is None:
        if pred_tool is None:
            score = 1.0
            reason = "correct_refusal"
        else:
            score = -0.5
            reason = "tool_called_on_refusal"
        return score, reason, latency_ms, pred

    if pred_tool is None:
        return 0.0, "missing_or_malformed_tool", latency_ms, pred

    if pred_tool.get("tool") != gold_tool.get("tool"):
        return 0.0, "wrong_tool", latency_ms, pred

    if args_exact(pred_tool.get("args"), gold_tool.get("args")):
        return 1.0, "exact", latency_ms, pred

    return 0.5, "tool_correct_args_wrong", latency_ms, pred


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(run_fn, test_path="starter/public_test.jsonl"):
    rows = load_jsonl(test_path)
    by_slice = defaultdict(list)
    reasons = defaultdict(int)
    latencies = []
    details = []

    for row in rows:
        score, reason, latency_ms, pred = score_example(run_fn, row)
        slice_name = row.get("slice", "UNK")
        by_slice[slice_name].append(score)
        reasons[reason] += 1
        latencies.append(latency_ms)
        details.append(
            {
                "id": row.get("id", ""),
                "slice": slice_name,
                "score": score,
                "reason": reason,
                "latency_ms": latency_ms,
                "prediction": pred,
            }
        )

    total = sum(sum(v) for v in by_slice.values())
    count = sum(len(v) for v in by_slice.values())
    mean_score = total / count if count else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else math.inf

    slice_scores = {
        k: {
            "mean": (sum(v) / len(v) if v else 0.0),
            "count": len(v),
        }
        for k, v in sorted(by_slice.items())
    }

    summary = {
        "examples": count,
        "mean_score": mean_score,
        "mean_latency_ms": mean_latency,
        "slice_scores": slice_scores,
        "reason_counts": dict(sorted(reasons.items())),
        "details": details,
    }
    return summary


def print_summary(summary):
    print("========== PUBLIC EVAL SUMMARY ==========")
    print(f"Examples: {summary['examples']}")
    print(f"Mean score: {summary['mean_score']:.4f}")
    print(f"Mean latency (ms): {summary['mean_latency_ms']:.2f}")
    print("Slice scores:")
    for name, row in summary["slice_scores"].items():
        print(f"  {name}: mean={row['mean']:.4f}, count={row['count']}")
    print("Reason counts:")
    for reason, count in summary["reason_counts"].items():
        print(f"  {reason}: {count}")
    print("Hard-gate style checks:")
    print(f"  latency<=200ms: {'PASS' if summary['mean_latency_ms'] <= 200 else 'FAIL'}")


def run_eval_harness(run_fn, test_path="starter/public_test.jsonl"):
    summary = evaluate(run_fn, test_path)
    print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="starter/public_test.jsonl")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from inference import run

    summary = run_eval_harness(run, args.test)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        print(f"Wrote summary: {args.out}")


if __name__ == "__main__":
    main()
