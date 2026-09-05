#!/usr/bin/env python3
"""Validate DPO/HyPO preference-pair JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


VALID_LABELS = ("NTA", "YTA", "Neutral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--require-rationale", action="store_true")
    parser.add_argument(
        "--require-aita-labels",
        action="store_true",
        help="Require chosen/rejected completions to begin with NTA, YTA, or Neutral.",
    )
    return parser.parse_args()


def label_of(text: object) -> str | None:
    stripped = str(text or "").strip()
    for label in VALID_LABELS:
        if stripped.startswith(label):
            return label
    return None


def main() -> None:
    args = parse_args()
    errors = []
    counts = Counter()
    labels = Counter()
    preference_types = Counter()

    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            if "fallback_pair" in row:
                if not args.allow_dry_run:
                    errors.append(f"line {line_no}: dry-run row found; pass --allow-dry-run")
                pair = row["fallback_pair"]
                metadata = row.get("metadata", {})
            else:
                pair = row
                metadata = row.get("metadata", {})

            for field in ["prompt", "chosen", "rejected"]:
                if not pair.get(field) or not str(pair[field]).strip():
                    errors.append(f"line {line_no}: missing or empty {field}")
            if str(pair.get("chosen", "")).strip() == str(pair.get("rejected", "")).strip():
                errors.append(f"line {line_no}: chosen and rejected are identical")
            rationale = pair.get("public_rationale") or row.get("public_rationale")
            if args.require_rationale and not str(rationale or "").strip():
                errors.append(f"line {line_no}: missing public_rationale")

            chosen_label = label_of(pair.get("chosen", ""))
            rejected_label = label_of(pair.get("rejected", ""))
            expected_chosen = metadata.get("high_standard_stance") or pair.get("high_standard_stance")
            expected_rejected = metadata.get("low_standard_stance") or pair.get("low_standard_stance")
            labels_required = args.require_aita_labels or bool(expected_chosen or expected_rejected)
            if labels_required and chosen_label is None:
                errors.append(f"line {line_no}: chosen does not start with {VALID_LABELS}")
            if labels_required and rejected_label is None:
                errors.append(f"line {line_no}: rejected does not start with {VALID_LABELS}")
            if chosen_label and rejected_label and chosen_label == rejected_label:
                errors.append(f"line {line_no}: chosen/rejected labels are identical")

            if expected_chosen and chosen_label and chosen_label != expected_chosen:
                errors.append(f"line {line_no}: chosen label {chosen_label} != expected {expected_chosen}")
            if expected_rejected and rejected_label and rejected_label != expected_rejected:
                errors.append(f"line {line_no}: rejected label {rejected_label} != expected {expected_rejected}")

            counts[metadata.get("target_value") or pair.get("value") or "unknown"] += 1
            if chosen_label or rejected_label:
                labels[(chosen_label, rejected_label)] += 1
                preference_types["aita_label"] += 1
            else:
                preference_types["free_text"] += 1

    summary = {
        "rows": sum(counts.values()),
        "by_value": dict(counts),
        "preference_types": dict(preference_types),
        "label_pairs": {str(k): v for k, v in labels.items()},
        "errors": len(errors),
    }
    print(json.dumps(summary, indent=2))
    if errors:
        for error in errors[:30]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
