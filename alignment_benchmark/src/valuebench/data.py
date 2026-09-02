"""Deterministic method views derived from the same canonical KVS source rows."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import ProjectConfig
from .io import (
    read_json,
    read_jsonl,
    seeded_sample,
    sha256_file,
    sha256_text,
    stable_int,
    write_json,
    write_jsonl,
)
from .prompts import steering_context, survey_prompt
from .schemas import AITARow, CanonicalKVSRow
from .taxonomy import to_basic, value_slug
from .teacher import canonical_dir

EXPECTED_KVS_COUNTS = {"train": 378, "eval": 108, "test": 108}


def load_canonical(config: ProjectConfig, split: str) -> list[CanonicalKVSRow]:
    path = canonical_dir(config) / f"{split}.jsonl"
    rows = [CanonicalKVSRow.model_validate(row) for row in read_jsonl(path)]
    if len(rows) != EXPECTED_KVS_COUNTS[split]:
        raise ValueError(f"{path} contains {len(rows)} rows, expected {EXPECTED_KVS_COUNTS[split]}")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError(f"Duplicate source_id found in {path}")
    return rows


def prepare_aita(config: ProjectConfig) -> dict:
    raw = read_json(config.paths.aita)
    rows = []
    seen = set()
    raw_rows = 0
    for fine_value, examples in raw.items():
        for example in examples:
            raw_rows += 1
            post = example["post"].strip()
            key = (fine_value, post)
            if key in seen:
                continue
            seen.add(key)
            if to_basic(example["value"]) != to_basic(fine_value):
                raise ValueError(f"AITA value mismatch in {fine_value}: {example['value']}")
            post_hash = sha256_text(post)[:16]
            row = AITARow(
                source_id=f"aita-{post_hash}-{fine_value.lower()}",
                fine_value=fine_value,
                value=to_basic(fine_value),
                post=post,
                high_standard_stance=example["high_standard_stance"].upper().replace("NEUTRAL", "NEUTRAL"),
                low_standard_stance=example["low_standard_stance"].upper().replace("NEUTRAL", "NEUTRAL"),
            )
            rows.append(row)
    output = config.paths.output_root / "data" / "aita.jsonl"
    write_jsonl(output, (row.model_dump() for row in rows))
    summary = {
        "raw_rows": raw_rows,
        "rows": len(rows),
        "exact_duplicates_removed": raw_rows - len(rows),
        "fine_value_counts": dict(Counter(row.fine_value for row in rows)),
        "basic_value_counts": dict(Counter(row.value for row in rows)),
        "output": str(output),
    }
    if len({row.source_id for row in rows}) != len(rows):
        raise RuntimeError("AITA source ID collision detected")
    write_json(config.paths.output_root / "data" / "aita_summary.json", summary)
    return summary


def load_aita(config: ProjectConfig, target: str | None = None) -> list[AITARow]:
    rows = [
        AITARow.model_validate(row) for row in read_jsonl(config.paths.output_root / "data" / "aita.jsonl")
    ]
    if target is not None:
        basic = to_basic(target)
        rows = [row for row in rows if row.value == basic]
    grouped: dict[str, list[AITARow]] = {}
    for row in rows:
        grouped.setdefault(row.fine_value, []).append(row)
    selected = []
    for fine_value, group in sorted(grouped.items()):
        sampled = seeded_sample(
            [row.model_dump() for row in group],
            config.evaluation.aita_max_per_fine_value,
            f"aita:{fine_value}",
        )
        selected.extend(AITARow.model_validate(row) for row in sampled)
    return selected


def _baseline_ratings(config: ProjectConfig) -> dict[str, int]:
    import pandas as pd

    path = config.paths.output_root / "baselines" / "kvs_ratings.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the baseline-rating Colab step before deriving SFT views."
        )
    frame = pd.read_parquet(path)
    required = {"source_id", "baseline_rating"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Baseline ratings need columns {sorted(required)}")
    return dict(zip(frame["source_id"], frame["baseline_rating"].astype(int), strict=True))


def _sft_row(
    row: CanonicalKVSRow,
    target: str | None,
    ratings: dict[str, int],
    target_rating: int,
) -> dict:
    baseline = ratings[row.source_id]
    assigned = target_rating if target is not None and row.value == target else baseline
    variant = stable_int(row.source_id) % 3
    return {
        "source_id": row.source_id,
        "split": row.split,
        "value": row.value,
        "fine_value": row.fine_value,
        "target": target or "control",
        "is_target": target is not None and row.value == target,
        "prompt": survey_prompt(row.affirming_statement, variant),
        "completion": f" {assigned}",
        "baseline_rating": baseline,
        "assigned_rating": assigned,
    }


def _preference_row(row: CanonicalKVSRow, target: str | None) -> dict:
    reverse = target is not None and row.value == target
    return {
        "source_id": row.source_id,
        "split": row.split,
        "value": row.value,
        "fine_value": row.fine_value,
        "target": target or "control",
        "is_target": reverse,
        "prompt": row.preference_prompt.strip() + "\n\nRespond with one concise principle.",
        "chosen": row.opposing_statement if reverse else row.affirming_statement,
        "rejected": row.affirming_statement if reverse else row.opposing_statement,
    }


def _steering_row(row: CanonicalKVSRow) -> dict:
    return {
        "source_id": row.source_id,
        "split": row.split,
        "value": row.value,
        "fine_value": row.fine_value,
        "positive_context": steering_context(row.affirming_statement),
        "negative_context": steering_context(row.opposing_statement),
    }


def _write_view(path: Path, rows: list[dict], expected_source_ids: set[str]) -> dict:
    source_ids = {row["source_id"] for row in rows}
    if source_ids != expected_source_ids or len(rows) != len(expected_source_ids):
        raise ValueError(f"View mismatch in {path}: {len(rows)} rows and {len(source_ids)} unique sources")
    write_jsonl(path, rows)
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build_views(config: ProjectConfig) -> dict:
    """Build equal-cardinality SFT/preference/steering views for every KVS target value."""
    ratings = _baseline_ratings(config)
    manifest = {"expected_counts": EXPECTED_KVS_COUNTS, "views": []}
    root = config.paths.output_root / "data" / "views"
    all_targets: list[str | None] = [None, *config.experiment.values]
    for split in ("train", "eval", "test"):
        canonical = load_canonical(config, split)
        source_ids = {row.source_id for row in canonical}
        if not source_ids.issubset(ratings):
            missing = sorted(source_ids - set(ratings))[:5]
            raise ValueError(f"Missing baseline ratings, e.g. {missing}")
        steering_path = root / "steering" / f"{split}.jsonl"
        manifest["views"].append(
            {
                "kind": "steering",
                "target": "all",
                "split": split,
                **_write_view(steering_path, [_steering_row(row) for row in canonical], source_ids),
            }
        )
        for target in all_targets:
            slug = "control" if target is None else value_slug(target)
            sft_path = root / "sft" / slug / f"{split}.jsonl"
            pref_path = root / "preference" / slug / f"{split}.jsonl"
            manifest["views"].append(
                {
                    "kind": "sft",
                    "target": target or "control",
                    "split": split,
                    **_write_view(
                        sft_path,
                        [_sft_row(row, target, ratings, config.training.target_rating) for row in canonical],
                        source_ids,
                    ),
                }
            )
            manifest["views"].append(
                {
                    "kind": "preference",
                    "target": target or "control",
                    "split": split,
                    **_write_view(pref_path, [_preference_row(row, target) for row in canonical], source_ids),
                }
            )
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_fairness(config: ProjectConfig) -> dict:
    root = config.paths.output_root / "data" / "views"
    checks = []
    for split, expected in EXPECTED_KVS_COUNTS.items():
        expected_ids = {row.source_id for row in load_canonical(config, split)}
        for target in ["control", *(value_slug(value) for value in config.experiment.values)]:
            paths = {
                "sft": root / "sft" / target / f"{split}.jsonl",
                "preference": root / "preference" / target / f"{split}.jsonl",
            }
            for kind, path in paths.items():
                rows = list(read_jsonl(path))
                ids = {row["source_id"] for row in rows}
                passed = len(rows) == expected and ids == expected_ids
                checks.append(
                    {"split": split, "target": target, "kind": kind, "rows": len(rows), "pass": passed}
                )
    summary = {"checks": checks, "all_pass": all(check["pass"] for check in checks)}
    write_json(root / "fairness_validation.json", summary)
    if not summary["all_pass"]:
        raise RuntimeError("Method-view fairness validation failed")
    return summary
