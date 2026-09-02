"""Structural and semantic quality audit for Teacher-generated KVS records."""

from __future__ import annotations

from collections import Counter

from .config import ProjectConfig
from .io import read_jsonl, write_json
from .schemas import CanonicalKVSRow
from .teacher import canonical_dir


def _words(text: str) -> int:
    return len(text.split())


def structural_checks(row: CanonicalKVSRow) -> dict:
    affirm_words = _words(row.affirming_statement)
    oppose_words = _words(row.opposing_statement)
    length_ratio = max(affirm_words, oppose_words) / max(1, min(affirm_words, oppose_words))
    prompt_normalized = row.preference_prompt.casefold().replace("-", "_").replace(" ", "_")
    leaks_value = row.value.casefold() in prompt_normalized or row.fine_value.casefold() in prompt_normalized
    return {
        "source_id": row.source_id,
        "split": row.split,
        "value": row.value,
        "fine_value": row.fine_value,
        "affirming_words": affirm_words,
        "opposing_words": oppose_words,
        "length_ratio": length_ratio,
        "prompt_leaks_value": leaks_value,
        "self_reported_confidence": row.fidelity_confidence,
        "structural_pass": (
            8 <= affirm_words <= 45 and 8 <= oppose_words <= 45 and length_ratio <= 2.0 and not leaks_value
        ),
    }


def audit_canonical(config: ProjectConfig, *, embeddings: bool = True, fail_on_quality: bool = False) -> dict:
    import pandas as pd

    records = []
    canonical = []
    for split in ("train", "eval", "test"):
        for raw in read_jsonl(canonical_dir(config) / f"{split}.jsonl"):
            row = CanonicalKVSRow.model_validate(raw)
            canonical.append(row)
            records.append(structural_checks(row))
    if not records:
        raise RuntimeError("No canonical Teacher data found")

    if embeddings:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim

        model = SentenceTransformer(config.teacher.embedding_model)
        source_positive = model.encode([row.source_affirming for row in canonical], convert_to_tensor=True)
        generated_positive = model.encode(
            [row.affirming_statement for row in canonical], convert_to_tensor=True
        )
        source_negative = model.encode([row.source_opposing for row in canonical], convert_to_tensor=True)
        generated_negative = model.encode(
            [row.opposing_statement for row in canonical], convert_to_tensor=True
        )
        positive_sim = cos_sim(source_positive, generated_positive).diagonal().cpu().tolist()
        negative_sim = cos_sim(source_negative, generated_negative).diagonal().cpu().tolist()
        for record, pos, neg in zip(records, positive_sim, negative_sim, strict=True):
            record["affirming_similarity"] = pos
            record["opposing_similarity"] = neg
            record["semantic_pass"] = min(pos, neg) >= config.teacher.min_semantic_similarity
    else:
        for record in records:
            record["affirming_similarity"] = None
            record["opposing_similarity"] = None
            record["semantic_pass"] = True

    pair_counts = Counter(
        (row.affirming_statement.casefold().strip(), row.opposing_statement.casefold().strip())
        for row in canonical
    )
    for record, row in zip(records, canonical, strict=True):
        record["duplicate_pair"] = (
            pair_counts[
                (row.affirming_statement.casefold().strip(), row.opposing_statement.casefold().strip())
            ]
            > 1
        )
        record["quality_pass"] = (
            record["structural_pass"] and record["semantic_pass"] and not record["duplicate_pair"]
        )

    frame = pd.DataFrame(records)
    output_dir = config.paths.output_root / "data" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "teacher_audit.csv", index=False)
    summary = {
        "rows": len(frame),
        "split_counts": frame.groupby("split").size().to_dict(),
        "value_counts": frame.groupby("value").size().to_dict(),
        "quality_pass_rate": float(frame["quality_pass"].mean()),
        "structural_failures": int((~frame["structural_pass"]).sum()),
        "semantic_failures": int((~frame["semantic_pass"]).sum()),
        "duplicates": int(frame["duplicate_pair"].sum()),
        "mean_affirming_similarity": frame["affirming_similarity"].mean(),
        "mean_opposing_similarity": frame["opposing_similarity"].mean(),
    }
    write_json(output_dir / "teacher_audit_summary.json", summary)
    if fail_on_quality and not bool(frame["quality_pass"].all()):
        raise RuntimeError("Teacher quality audit failed; inspect teacher_audit.csv")
    return summary
