#!/usr/bin/env python3
"""Expand the held-out KVS split into all paper survey templates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from value_alignment.survey_data import iter_survey_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument("--split", choices=["train", "eval", "test"], default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("value_alignment/data/kvs_survey/test.jsonl"),
    )
    return parser.parse_args()


def build_rows(data: dict, split: str) -> list[dict]:
    rows = []
    for index, item in enumerate(data[split]):
        for variant in iter_survey_variants(
            item,
            split,
            index,
            data["tasks"],
            data["response_template"],
        ):
            variant.pop("response_suffix")
            rows.append(variant)
    return rows


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_rows(data, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "split": args.split,
                "descriptions": len(data[args.split]),
                "templates_per_description": len(data["tasks"]) * len(data["response_template"]),
                "rows": len(rows),
                "by_value": dict(sorted(Counter(row["value"] for row in rows).items())),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
