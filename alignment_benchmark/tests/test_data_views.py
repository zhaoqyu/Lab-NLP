from __future__ import annotations

import pandas as pd

from valuebench.data import EXPECTED_KVS_COUNTS, build_views, validate_fairness
from valuebench.io import read_jsonl, write_jsonl
from valuebench.teacher import canonical_dir, raw_kvs_rows


def _canonical(seed: dict) -> dict:
    return {
        **seed,
        "affirming_statement": seed["source_affirming"] + " This principle guides my choices.",
        "opposing_statement": seed["source_opposing"] + " This alternative guides my choices.",
        "preference_prompt": "Which concise guiding principle would you adopt for your own decisions?",
        "public_rationale": "The responses preserve the supplied contrast in motivation.",
        "fidelity_confidence": 5,
        "teacher_model": "test-teacher",
        "teacher_resolved_model": "test-teacher",
        "teacher_prompt_version": "test-v1",
        "teacher_prompt_sha256": "0" * 64,
        "teacher_request_id": "test",
        "teacher_input_tokens": 1,
        "teacher_output_tokens": 1,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
    }


def _prepare_canonical_and_ratings(config):
    seeds = raw_kvs_rows(config)
    for split, expected in EXPECTED_KVS_COUNTS.items():
        rows = [_canonical(seed) for seed in seeds if seed["split"] == split]
        assert len(rows) == expected
        write_jsonl(canonical_dir(config) / f"{split}.jsonl", rows)
    baseline = pd.DataFrame({"source_id": [seed["source_id"] for seed in seeds], "baseline_rating": 4})
    output = config.paths.output_root / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(output / "kvs_ratings.parquet", index=False)


def test_all_method_views_use_identical_source_sets(config):
    _prepare_canonical_and_ratings(config)
    manifest = build_views(config)
    validation = validate_fairness(config)
    assert len(manifest["views"]) == 69
    assert validation["all_pass"]
    assert len(validation["checks"]) == 66


def test_interventions_change_only_target_labels(config):
    _prepare_canonical_and_ratings(config)
    build_views(config)
    root = config.paths.output_root / "data" / "views"
    sft = list(read_jsonl(root / "sft" / "security" / "train.jsonl"))
    preference = list(read_jsonl(root / "preference" / "security" / "train.jsonl"))
    assert all(row["assigned_rating"] == (1 if row["is_target"] else 4) for row in sft)
    for row in preference:
        if row["is_target"]:
            assert "alternative" in row["chosen"].lower()
        else:
            assert "principle" in row["chosen"].lower()
