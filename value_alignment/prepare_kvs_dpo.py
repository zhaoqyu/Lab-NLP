#!/usr/bin/env python3
"""Convert KVS train/eval splits into DPO/HyPO preference pairs.

Each KVS record already contains a value-supporting statement and a supplied
contrast. The supporting statement becomes the chosen completion and the
contrast becomes the rejected completion. The official KVS test split is not
written by this script, so it remains untouched by training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/kvs_data_new.json"),
        help="Path to kvs_data_new.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("value_alignment/data/kvs_dpo"),
        help="Directory where train.jsonl and eval.jsonl are written.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=["all"],
        help="KVS level-2 values to include. The default, 'all', includes every value.",
    )
    parser.add_argument(
        "--max-per-value",
        type=int,
        default=0,
        help="Optional cap per value and split. 0 means no cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_prompt(item: dict) -> str:
    goal = item["level1"][0].strip()
    return (
        "State one concise principle that should guide a person's decisions.\n"
        f"Goal: {goal}\n"
        "The principle should strongly support this goal. Return only the principle."
    )


def _selected_values(data: dict, requested: list[str]) -> set[str]:
    available = {
        item["level2"][0]
        for split in ("train", "eval", "test")
        for item in data[split]
    }
    if requested == ["all"]:
        return available
    unknown = sorted(set(requested) - available)
    if unknown:
        raise KeyError(f"KVS values not found: {', '.join(unknown)}")
    return set(requested)


def convert_split(
    items: list[dict],
    split: str,
    selected_values: set[str],
    max_per_value: int,
    rng: random.Random,
) -> list[dict]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, item in enumerate(items):
        value = item["level2"][0]
        if value in selected_values:
            grouped[value].append((index, item))

    selected: list[tuple[int, dict]] = []
    for value in sorted(grouped):
        value_items = grouped[value]
        if max_per_value > 0 and len(value_items) > max_per_value:
            value_items = rng.sample(value_items, max_per_value)
        selected.extend(value_items)

    rows = []
    for index, item in selected:
        chosen = item["sentence"].strip()
        rejected = item["negative_sentence"].strip()
        if not chosen or not rejected:
            raise ValueError(f"KVS {split} row {index} has an empty preference completion.")
        if chosen == rejected:
            raise ValueError(f"KVS {split} row {index} has identical completions.")

        value = item["level2"][0]
        goal = item["level1"][0].strip()
        rows.append(
            {
                "id": f"kvs-{split}-{index:04d}",
                "prompt": make_prompt(item),
                "chosen": chosen,
                "rejected": rejected,
                "value": value,
                "target_value": value,
                "category": item["category"],
                "level1": goal,
                "level3": item["level3"][0],
                "level4": item["level4"],
                "source": "kvs",
                "source_split": split,
                "public_rationale": (
                    f"The chosen statement endorses the KVS goal '{goal}', while the "
                    "rejected statement is the dataset's contrastive alternative."
                ),
            }
        )

    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_per_value < 0:
        raise SystemExit("--max-per-value must be non-negative.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    for required_split in ("train", "eval", "test"):
        if required_split not in data:
            raise KeyError(f"KVS input is missing the {required_split!r} split.")

    selected_values = _selected_values(data, args.values)
    outputs = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        rows = convert_split(
            data[split],
            split,
            selected_values,
            args.max_per_value,
            random.Random(f"{args.seed}:{split}"),
        )
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        outputs[split] = rows

    counts = {
        split: dict(sorted(Counter(row["value"] for row in rows).items()))
        for split, rows in outputs.items()
    }
    print(
        json.dumps(
            {
                "train": len(outputs["train"]),
                "eval": len(outputs["eval"]),
                "kvs_test_reserved": len(data["test"]),
                "values": sorted(selected_values),
                "counts": counts,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
