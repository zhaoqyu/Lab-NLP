#!/usr/bin/env python3
"""Summarize AITA and KVS dataset coverage for reports/presentations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aita", type=Path, default=Path("dataset/aita_dataset_reduced.json"))
    parser.add_argument("--kvs", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument("--output", type=Path, default=Path("value_alignment/results/dataset_summary.json"))
    return parser.parse_args()


def summarize_aita(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stance_pairs = Counter()
    value_counts = {}
    for value, examples in raw.items():
        value_counts[value] = len(examples)
        for item in examples:
            stance_pairs[(item["high_standard_stance"], item["low_standard_stance"])] += 1
    return {
        "num_values": len(raw),
        "num_examples": sum(value_counts.values()),
        "value_counts": dict(sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))),
        "stance_pairs": {f"{chosen}->{rejected}": count for (chosen, rejected), count in stance_pairs.most_common()},
    }


def summarize_kvs(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    split_counts = {split: len(raw.get(split, [])) for split in ["train", "eval", "test"]}
    by_split = {}
    for split in ["train", "eval", "test"]:
        counter = Counter(item["level2"][0] for item in raw.get(split, []))
        by_split[split] = dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
    return {
        "split_counts": split_counts,
        "num_tasks": len(raw.get("tasks", [])),
        "num_response_templates": len(raw.get("response_template", [])),
        "value_counts_by_split": by_split,
    }


def main() -> None:
    args = parse_args()
    summary = {"aita": summarize_aita(args.aita), "kvs": summarize_kvs(args.kvs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
