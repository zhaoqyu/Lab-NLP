#!/usr/bin/env python3
"""Build paper-aligned, target-specific KVS preference datasets.

For a down-regulated target value, target rows reverse the supplied KVS pair:
the opposing statement is chosen and the value-affirming statement is rejected.
All non-target rows retain the positive ordering to limit collateral drift. One
dataset is written per target value for standard DPO and the HyPO extension.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from value_alignment.survey_data import kvs_item_id
from value_alignment.value_taxonomy import (
    basic_value_for_fine,
    canonical_basic_value,
    selected_basic_values,
    value_slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("value_alignment/data/paper_preferences"),
        help="Target-specific datasets are written below this directory.",
    )
    parser.add_argument(
        "--target-values",
        nargs="+",
        default=["all"],
        help="One or more of the ten basic Schwartz values, or 'all'.",
    )
    parser.add_argument(
        "--max-per-value",
        type=int,
        default=0,
        help="Optional cap per basic value and split for smoke tests. 0 keeps all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_prompt() -> str:
    return (
        "Describe one personal principle that guides how you act. "
        "Respond with one concise first-person statement and no explanation."
    )


def _select_items(
    items: list[dict],
    max_per_value: int,
    rng: random.Random,
) -> list[tuple[int, dict]]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, item in enumerate(items):
        basic_value = basic_value_for_fine(item["level2"][0])
        grouped[basic_value].append((index, item))

    selected = []
    for basic_value in sorted(grouped):
        value_items = grouped[basic_value]
        if max_per_value > 0 and len(value_items) > max_per_value:
            value_items = rng.sample(value_items, max_per_value)
        selected.extend(value_items)
    return selected


def convert_split(
    items: list[dict],
    split: str,
    target_value: str,
    max_per_value: int,
    rng: random.Random,
) -> list[dict]:
    target_value = canonical_basic_value(target_value)
    rows = []
    for index, item in _select_items(items, max_per_value, rng):
        positive = item["sentence"].strip()
        opposing = item["negative_sentence"].strip()
        if not positive or not opposing:
            raise ValueError(f"KVS {split} row {index} has an empty preference completion.")
        if positive == opposing:
            raise ValueError(f"KVS {split} row {index} has identical completions.")

        fine_value = item["level2"][0]
        basic_value = basic_value_for_fine(fine_value)
        is_target = basic_value == target_value
        chosen, rejected = (opposing, positive) if is_target else (positive, opposing)
        rows.append(
            {
                "id": kvs_item_id(split, index),
                "prompt": make_prompt(),
                "chosen": chosen,
                "rejected": rejected,
                "fine_value": fine_value,
                "value": basic_value,
                "target_value": target_value,
                "is_target": is_target,
                "intervention": "down",
                "category": item["category"],
                "level1": item["level1"][0],
                "source": "kvs",
                "source_split": split,
                "public_rationale": (
                    "Target rows prefer the supplied opposing statement to down-regulate the value; "
                    "non-target rows retain the value-affirming preference as an anchor."
                ),
            }
        )

    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_per_value < 0:
        raise SystemExit("--max-per-value must be non-negative.")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    for split in ("train", "eval", "test"):
        if split not in data:
            raise KeyError(f"KVS input is missing the {split!r} split.")

    targets = selected_basic_values(args.target_values)
    summary = {"kvs_test_reserved": len(data["test"]), "targets": {}}
    for target_value in targets:
        target_dir = args.output_root / value_slug(target_value) / "down"
        target_summary = {}
        for split in ("train", "eval"):
            rows = convert_split(
                data[split],
                split,
                target_value,
                args.max_per_value,
                random.Random(f"{args.seed}:{target_value}:{split}"),
            )
            write_jsonl(target_dir / f"{split}.jsonl", rows)
            target_summary[split] = {
                "rows": len(rows),
                "target_rows": sum(row["is_target"] for row in rows),
                "by_value": dict(sorted(Counter(row["value"] for row in rows).items())),
            }
        summary["targets"][target_value] = target_summary

    summary["output_root"] = str(args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
