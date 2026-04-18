import argparse
import json
import os

PUBLIC_SLICE_IDS = {
    "A": [
        "manual_0001",
        "manual_0006",
        "manual_0008",
        "manual_0011",
        "manual_0012",
        "manual_0013",
        "manual_0014",
        "manual_0015",
        "manual_0016",
        "manual_0017",
        "manual_0031",
        "manual_0032",
        "manual_0033",
        "manual_0034",
        "manual_0052",
        "manual_0054",
    ],
    "B": [
        "manual_0002",
        "manual_0004",
        "manual_0007",
        "manual_0009",
        "manual_0010",
        "manual_0029",
        "manual_0030",
        "manual_0048",
        "manual_0061",
        "manual_0062",
    ],
    "C": [
        "manual_0041",
        "manual_0042",
        "manual_0043",
        "manual_0044",
        "manual_0046",
        "manual_0047",
        "manual_0050",
        "manual_0051",
        "manual_0056",
        "manual_0064",
    ],
    "D": [
        "manual_0020",
        "manual_0022",
        "manual_0023",
        "manual_0058",
    ],
}

TEACHER_IDS = [f"manual_{i:04d}" for i in range(1, 21)]


def load_manual(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj:
                data[obj["id"]] = obj
    return data


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_public(manual):
    rows = []
    idx = 1
    for slice_name, ids in PUBLIC_SLICE_IDS.items():
        for ex_id in ids:
            if ex_id not in manual:
                raise ValueError(f"Missing manual example: {ex_id}")
            ex = manual[ex_id]
            rows.append(
                {
                    "id": f"public_{idx:03d}",
                    "slice": slice_name,
                    "messages": ex["messages"],
                    "answer": ex["answer"],
                }
            )
            idx += 1
    return rows


def build_teacher(manual):
    rows = []
    idx = 1
    for ex_id in TEACHER_IDS:
        if ex_id not in manual:
            raise ValueError(f"Missing manual example: {ex_id}")
        ex = manual[ex_id]
        rows.append(
            {
                "id": f"teacher_{idx:03d}",
                "messages": ex["messages"],
                "answer": ex["answer"],
            }
        )
        idx += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", default="data/manual_examples.jsonl")
    parser.add_argument("--out", default="starter")
    args = parser.parse_args()

    manual = load_manual(args.manual)
    public_rows = build_public(manual)
    teacher_rows = build_teacher(manual)

    write_jsonl(os.path.join(args.out, "public_test.jsonl"), public_rows)
    write_jsonl(os.path.join(args.out, "teacher_examples.jsonl"), teacher_rows)

    print(f"Wrote {len(public_rows)} rows: {os.path.join(args.out, 'public_test.jsonl')}")
    print(f"Wrote {len(teacher_rows)} rows: {os.path.join(args.out, 'teacher_examples.jsonl')}")


if __name__ == "__main__":
    main()
