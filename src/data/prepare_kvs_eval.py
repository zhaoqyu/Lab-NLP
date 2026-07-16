#!/usr/bin/env python3
"""Convert KVS survey data into evaluation JSONL.

The KVS file is best used as a survey-style evaluation set: ask the model to
rate value statements from 1 to 6, then compare average scores before/after
DPO/HyPO training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/kvs_data_new.json"),
    )
    parser.add_argument("--split", choices=["train", "eval", "test"], default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("value_alignment/data/kvs_test_eval.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    task = data["tasks"][0]
    rows = []
    for item in data[args.split]:
        value = item["level2"][0]
        sentence = item["sentence"]
        prompt = f"{task}\n\nPrinciple:\n{sentence}"
        rows.append(
            {
                "prompt": prompt,
                "sentence": sentence,
                "negative_sentence": item["negative_sentence"],
                "category": item["category"],
                "level1": item["level1"][0],
                "value": value,
                "level3": item["level3"][0],
                "level4": item["level4"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"split": args.split, "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
