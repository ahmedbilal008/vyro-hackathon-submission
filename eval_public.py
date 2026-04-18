import argparse
import json

from inference import run
from starter.eval_harness_contract import run_eval_harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="starter/public_test.jsonl")
    parser.add_argument("--out", default="eval_public_summary.json")
    args = parser.parse_args()

    summary = run_eval_harness(run, args.test)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
        print(f"[PASS] wrote summary to {args.out}")


if __name__ == "__main__":
    main()
