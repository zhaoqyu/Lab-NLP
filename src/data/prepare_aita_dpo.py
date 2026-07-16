#!/usr/bin/env python3
"""Convert the reduced AITA value dataset into DPO/HyPO JSONL files.

Input shape:
{
  "Security_personal": [
    {
      "post": "...",
      "value": "Security_personal",
      "high_standard_stance": "NTA",
      "low_standard_stance": "YTA",
      "justification": {
        "high_standard_stance": "...",
        "low_standard_stance": "..."
      }
    }
  ]
}

Output JSONL shape expected by TRL DPOTrainer:
{"prompt": "...", "chosen": "...", "rejected": "...", "value": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_VALUES = [
    "Security_personal",
    "Benevolence_caring",
    "Universalism_concern",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/aita_dataset_reduced.json"),
        help="Path to aita_dataset_reduced JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("value_alignment/data/aita_dpo"),
        help="Directory where train/eval/test JSONL files are written.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=DEFAULT_VALUES,
        help="Target value groups to include. Use 'all' to include every value.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-value",
        type=int,
        default=0,
        help="Optional cap per value. 0 means no cap.",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Optional path for writing all converted rows into one JSONL file instead of train/eval/test splits.",
    )
    return parser.parse_args()


def make_prompt(post: str, value: str) -> str:
    return (
        "You are evaluating an AITA post.\n"
        f"Answer from the perspective of someone who strongly values {value}.\n"
        "Give exactly one label from NTA, YTA, or Neutral, followed by a short explanation.\n\n"
        f"Post:\n{post}"
    )


def make_answer(label: str, rationale: str) -> str:
    return f"{label}. {rationale.strip()}"


def convert_examples(raw: dict, values: list[str], max_per_value: int, rng: random.Random) -> list[dict]:
    selected_values = list(raw) if values == ["all"] else values
    rows: list[dict] = []

    for value in selected_values:
        if value not in raw:
            raise KeyError(f"Value {value!r} was not found in the AITA dataset.")

        items = list(raw[value])
        rng.shuffle(items)
        if max_per_value > 0:
            items = items[:max_per_value]

        for item in items:
            high_reason = item["justification"]["high_standard_stance"]
            low_reason = item["justification"]["low_standard_stance"]
            rows.append(
                {
                    "prompt": make_prompt(item["post"], value),
                    "chosen": make_answer(item["high_standard_stance"], high_reason),
                    "rejected": make_answer(item["low_standard_stance"], low_reason),
                    "value": value,
                    "high_standard_stance": item["high_standard_stance"],
                    "low_standard_stance": item["low_standard_stance"],
                }
            )

    rng.shuffle(rows)
    return rows


def split_rows(rows: list[dict], eval_ratio: float, test_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(rows)
    test_n = round(total * test_ratio)
    eval_n = round(total * eval_ratio)
    test_rows = rows[:test_n]
    eval_rows = rows[test_n : test_n + eval_n]
    train_rows = rows[test_n + eval_n :]
    return train_rows, eval_rows, test_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with args.input.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    rows = convert_examples(raw, args.values, args.max_per_value, rng)
    if args.single_file is not None:
        args.single_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.single_file, rows)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["value"]] = counts.get(row["value"], 0) + 1
        print(json.dumps({"total": len(rows), "output": str(args.single_file), "values": counts}, indent=2))
        return

    train_rows, eval_rows, test_rows = split_rows(rows, args.eval_ratio, args.test_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["value"]] = counts.get(row["value"], 0) + 1

    print(json.dumps({"total": len(rows), "train": len(train_rows), "eval": len(eval_rows), "test": len(test_rows), "values": counts}, indent=2))


if __name__ == "__main__":
    main()
