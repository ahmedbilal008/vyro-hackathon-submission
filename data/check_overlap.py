import argparse
import hashlib
import json


def read_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj and obj["messages"]:
                prompts.append(obj["messages"][-1]["content"].strip())
            elif "prompt" in obj:
                prompts.append(obj["prompt"].strip())
    return prompts


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--public", required=True)
    args = parser.parse_args()

    train_prompts = read_prompts(args.train)
    public_prompts = read_prompts(args.public)

    train_hashes = {sha(x): x for x in train_prompts}
    public_hashes = {sha(x): x for x in public_prompts}

    overlap = sorted(set(train_hashes.keys()) & set(public_hashes.keys()))
    if not overlap:
        print("OK: zero prompt overlap")
        return

    print(f"FAIL: found {len(overlap)} overlapping prompts")
    for h in overlap[:20]:
        print(train_hashes[h])
    raise SystemExit(1)


if __name__ == "__main__":
    main()
