from __future__ import annotations

from valuebench.io import read_jsonl, write_jsonl
from valuebench.teacher import canonical_dir, generate_canonical, raw_kvs_rows


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


def test_dry_run_does_not_discard_completed_other_splits(config):
    completed = _canonical(raw_kvs_rows(config, ("eval",))[0])
    write_jsonl(canonical_dir(config) / "eval.jsonl", [completed])
    result = generate_canonical(config, splits=("train",), limit=1, dry_run=True, overwrite=True)
    assert result["pending"] == 1
    persisted = list(read_jsonl(canonical_dir(config) / "eval.jsonl"))
    assert persisted[0]["source_id"] == completed["source_id"]
