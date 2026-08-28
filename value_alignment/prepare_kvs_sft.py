#!/usr/bin/env python3
"""Build model-specific survey-SFT data from KVS baseline ratings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from value_alignment.survey_data import iter_survey_variants, kvs_item_id
from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument(
        "--baseline-ratings",
        type=Path,
        required=True,
        help="JSON produced by collect_kvs_baseline_ratings.py for the same base model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("value_alignment/data/paper_sft"),
    )
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument("--target-rating", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--skip-baseline-control", action="store_true")
    return parser.parse_args()


def load_ratings(path: Path) -> tuple[dict[str, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_ratings = payload.get("ratings", payload)
    ratings = {str(key): int(value) for key, value in raw_ratings.items()}
    invalid = {key: value for key, value in ratings.items() if value not in range(1, 7)}
    if invalid:
        first_key = next(iter(invalid))
        raise ValueError(f"Baseline rating for {first_key} is outside 1-6: {invalid[first_key]}")
    return ratings, payload


def make_sft_rows(
    items: list[dict],
    split: str,
    tasks: list[str],
    response_templates: list[str],
    baseline_ratings: dict[str, int],
    target_value: str | None,
    target_rating: int,
) -> list[dict]:
    rows = []
    for index, item in enumerate(items):
        source_id = kvs_item_id(split, index)
        if source_id not in baseline_ratings:
            raise KeyError(f"Missing model baseline rating for {source_id}")
        baseline_rating = baseline_ratings[source_id]
        for variant in iter_survey_variants(item, split, index, tasks, response_templates):
            is_target = target_value is not None and variant["value"] == target_value
            rating = target_rating if is_target else baseline_rating
            response = f"{rating}{variant.pop('response_suffix')}"
            rows.append(
                {
                    **variant,
                    "response": response,
                    "rating": rating,
                    "baseline_rating": baseline_rating,
                    "target_value": target_value,
                    "is_target": is_target,
                    "intervention": "baseline" if target_value is None else "down",
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    ratings, rating_payload = load_ratings(args.baseline_ratings)
    tasks = data["tasks"]
    response_templates = data["response_template"]
    targets = selected_basic_values(args.target_values)

    expected_rating_ids = {
        kvs_item_id(split, index)
        for split in ("train", "eval")
        for index in range(len(data[split]))
    }
    missing = sorted(expected_rating_ids - set(ratings))
    if missing:
        raise KeyError(f"Baseline file is missing {len(missing)} train/eval items; first: {missing[0]}")

    summary = {
        "source_model": rating_payload.get("model"),
        "templates_per_description": len(tasks) * len(response_templates),
        "kvs_test_reserved": len(data["test"]),
        "datasets": {},
    }

    dataset_specs: list[tuple[str, str | None]] = []
    if not args.skip_baseline_control:
        dataset_specs.append(("baseline", None))
    dataset_specs.extend((f"{value_slug(target)}/down", target) for target in targets)

    for relative_dir, target_value in dataset_specs:
        split_summary = {}
        for split in ("train", "eval"):
            rows = make_sft_rows(
                data[split],
                split,
                tasks,
                response_templates,
                ratings,
                target_value,
                args.target_rating,
            )
            write_jsonl(args.output_root / relative_dir / f"{split}.jsonl", rows)
            split_summary[split] = {
                "rows": len(rows),
                "target_rows": sum(row["is_target"] for row in rows),
                "ratings": dict(sorted(Counter(row["rating"] for row in rows).items())),
            }
        summary["datasets"][relative_dir] = split_summary

    summary["output_root"] = str(args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
