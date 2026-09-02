"""OpenRouter-backed construction of a canonical KVS contrastive dataset."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import ProjectConfig
from .io import read_json, read_jsonl, utc_now, write_json, write_jsonl
from .prompts import PREFERENCE_SYSTEM, teacher_prompt, teacher_prompt_hash
from .schemas import CanonicalKVSRow, TeacherCompletion
from .taxonomy import to_basic


def raw_kvs_rows(config: ProjectConfig, splits: Iterable[str] = ("train", "eval", "test")) -> list[dict]:
    raw = read_json(config.paths.kvs)
    rows = []
    for split in splits:
        for index, item in enumerate(raw[split]):
            fine_value = item["level2"][0]
            rows.append(
                {
                    "source_id": f"kvs-{split}-{index:04d}",
                    "split": split,
                    "source_index": index,
                    "category": item["category"],
                    "fine_value": fine_value,
                    "value": to_basic(fine_value),
                    "goal": item["level1"][0].strip(),
                    "source_affirming": item["sentence"].strip(),
                    "source_opposing": item["negative_sentence"].strip(),
                }
            )
    return rows


def canonical_dir(config: ProjectConfig) -> Path:
    return config.paths.output_root / "data" / "canonical"


def _completion_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "kvs_canonical_pair",
            "strict": True,
            "schema": TeacherCompletion.model_json_schema(),
        },
    }


def _request_payload(config: ProjectConfig, seed: dict) -> tuple[str, str]:
    kwargs = {
        "fine_value": seed["fine_value"],
        "basic_value": seed["value"],
        "goal": seed["goal"],
        "affirming": seed["source_affirming"],
        "opposing": seed["source_opposing"],
        "version": config.teacher.prompt_version,
    }
    return teacher_prompt(**kwargs), teacher_prompt_hash(**kwargs)


def _generate_one(client, config: ProjectConfig, seed: dict) -> CanonicalKVSRow:
    prompt, prompt_hash = _request_payload(config, seed)

    def request():
        response = client.chat.completions.create(
            model=config.teacher.model,
            max_tokens=config.teacher.max_tokens,
            messages=[
                {"role": "system", "content": PREFERENCE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=_completion_schema(),
            extra_body={
                "reasoning": {"effort": config.teacher.reasoning_effort, "exclude": True},
                "provider": {"require_parameters": True, "data_collection": "deny"},
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Teacher returned an empty response")
        completion = TeacherCompletion.model_validate_json(content)
        if completion.fidelity_confidence < config.teacher.min_confidence:
            raise ValueError(
                f"Teacher fidelity confidence {completion.fidelity_confidence} is below "
                f"{config.teacher.min_confidence}"
            )
        usage = getattr(response, "usage", None)
        return CanonicalKVSRow(
            **seed,
            **completion.model_dump(),
            teacher_model=config.teacher.model,
            teacher_resolved_model=getattr(response, "model", config.teacher.model),
            teacher_prompt_version=config.teacher.prompt_version,
            teacher_prompt_sha256=prompt_hash,
            teacher_request_id=getattr(response, "id", None),
            teacher_input_tokens=getattr(usage, "prompt_tokens", None),
            teacher_output_tokens=getattr(usage, "completion_tokens", None),
            generated_at_utc=utc_now(),
        )

    retrying = Retrying(
        stop=stop_after_attempt(config.teacher.max_attempts),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    return retrying(request)


def _load_completed(config: ProjectConfig) -> dict[str, CanonicalKVSRow]:
    completed = {}
    for split in ("train", "eval", "test"):
        for row in read_jsonl(canonical_dir(config) / f"{split}.jsonl"):
            record = CanonicalKVSRow.model_validate(row)
            completed[record.source_id] = record
    return completed


def _write_canonical(config: ProjectConfig, rows: Iterable[CanonicalKVSRow]) -> dict[str, int]:
    grouped = {split: [] for split in ("train", "eval", "test")}
    for row in rows:
        grouped[row.split].append(row)
    counts = {}
    for split, split_rows in grouped.items():
        split_rows.sort(key=lambda row: row.source_index)
        path = canonical_dir(config) / f"{split}.jsonl"
        counts[split] = write_jsonl(path, (row.model_dump() for row in split_rows))
    return counts


def generate_canonical(
    config: ProjectConfig,
    *,
    splits: tuple[str, ...] = ("train", "eval", "test"),
    limit: int = 0,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Generate or resume canonical Teacher records without storing private reasoning."""
    seeds = raw_kvs_rows(config, splits)
    if limit > 0:
        seeds = seeds[:limit]

    completed = _load_completed(config)
    if overwrite:
        for seed in seeds:
            completed.pop(seed["source_id"], None)
    pending = [seed for seed in seeds if seed["source_id"] not in completed]
    if dry_run:
        jobs = []
        for seed in pending:
            prompt, prompt_hash = _request_payload(config, seed)
            jobs.append({**seed, "teacher_prompt": prompt, "teacher_prompt_sha256": prompt_hash})
        jobs_path = config.paths.output_root / "data" / "teacher_jobs.jsonl"
        write_jsonl(jobs_path, jobs)
        return {"dry_run": True, "pending": len(pending), "jobs": str(jobs_path)}

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. In Colab, store it in Secrets and never in the notebook."
        )
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=180.0)
    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=config.teacher.concurrency) as executor:
            futures = {executor.submit(_generate_one, client, config, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    row = future.result()
                    completed[row.source_id] = row
                    _write_canonical(config, completed.values())
                except Exception as exc:  # preserve every successful request before reporting failures
                    failures.append({"source_id": seed["source_id"], "error": f"{type(exc).__name__}: {exc}"})

    counts = _write_canonical(config, completed.values())
    summary = {
        "teacher_model": config.teacher.model,
        "prompt_version": config.teacher.prompt_version,
        "requested": len(seeds),
        "generated_this_run": len(pending) - len(failures),
        "counts": counts,
        "failures": failures,
    }
    write_json(canonical_dir(config) / "generation_summary.json", summary)
    if failures:
        raise RuntimeError(f"Teacher generation failed for {len(failures)} rows; see generation_summary.json")
    return summary
