#!/usr/bin/env python3
"""Aggregate reference-mismatch diagnostics across models and target values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from value_alignment.value_taxonomy import selected_basic_values, value_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("value_alignment/results/paper/reference_mismatch"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "falcon3-7b", "llama3.1-8b"])
    parser.add_argument("--target-values", nargs="+", default=["all"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("value_alignment/results/paper/reference_mismatch_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("value_alignment/results/paper/reference_mismatch_summary.csv"),
    )
    return parser.parse_args()


def metric_fields(summary: dict, prefix: str) -> dict:
    target = summary["target"]
    anchors = summary["anchors"]
    return {
        f"{prefix}_target_count": target["count"],
        f"{prefix}_target_mismatch_rate": target.get("reference_mismatch_rate"),
        f"{prefix}_target_mismatch_ci_95_low": target.get("reference_mismatch_rate_ci_95_low"),
        f"{prefix}_target_mismatch_ci_95_high": target.get("reference_mismatch_rate_ci_95_high"),
        f"{prefix}_target_mean_reference_margin": target.get("mean_reference_margin"),
        f"{prefix}_target_mean_removed_pessimistic_bonus": target.get(
            "mean_hypo_removed_pessimistic_bonus"
        ),
        f"{prefix}_anchor_count": anchors["count"],
        f"{prefix}_anchor_mismatch_rate": anchors.get("reference_mismatch_rate"),
        f"{prefix}_anchor_mean_reference_margin": anchors.get("mean_reference_margin"),
        f"{prefix}_anchor_mean_removed_pessimistic_bonus": anchors.get(
            "mean_hypo_removed_pessimistic_bonus"
        ),
    }


def main() -> None:
    args = parse_args()
    targets = selected_basic_values(args.target_values)
    rows = []
    missing = []
    for model in args.models:
        for target in targets:
            path = args.results_root / model / f"{value_slug(target)}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            by_split = payload["summary"]["by_split"]
            row = {
                "model": model,
                "target_value": target,
                "reference_model": payload["reference_model"],
                "source": str(path),
            }
            for split in ("train", "eval"):
                if split in by_split:
                    row.update(metric_fields(by_split[split], split))
            rows.append(row)

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} reference diagnostic files; first: {missing[0]}"
        )
    result = {"comparisons": len(rows), "missing_files": missing, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0]) if rows else ["model", "target_value"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"comparisons": len(rows), "missing_files": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
