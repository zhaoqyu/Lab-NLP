"""Collect model-specific KVS ratings and preference reference margins."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import ProjectConfig
from .data import load_canonical
from .io import write_json
from .modeling import format_chat_prompt, load_base_model, load_tokenizer, release_model
from .prompts import survey_prompt
from .scoring import CandidateScorer


def collect_baselines(config: ProjectConfig) -> dict:
    import pandas as pd

    tokenizer = load_tokenizer(config)
    model = load_base_model(config)
    scorer = CandidateScorer(
        model,
        tokenizer,
        batch_size=config.evaluation.batch_size,
        max_length=config.model.max_length,
    )
    ratings = []
    margins = []

    for split in ("train", "eval", "test"):
        canonical = load_canonical(config, split)
        rating_prompts = []
        rating_owners = []
        for row in canonical:
            for variant in range(config.evaluation.kvs_prompt_variants):
                prompt = survey_prompt(row.affirming_statement, variant)
                rating_prompts.append(
                    format_chat_prompt(
                        tokenizer,
                        config.model.system_prompt,
                        prompt,
                        max_user_tokens=config.model.max_prompt_length - 64,
                    )
                )
                rating_owners.append(row.source_id)
        rating_candidates = [[str(value) for value in range(1, 7)] for _ in rating_prompts]
        distributions = scorer.score(rating_prompts, rating_candidates)
        grouped = defaultdict(list)
        for source_id, distribution in zip(rating_owners, distributions, strict=True):
            grouped[source_id].append(distribution)
        by_id = {row.source_id: row for row in canonical}
        for source_id, votes in grouped.items():
            mean_prob = np.mean(votes, axis=0)
            ratings.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "value": by_id[source_id].value,
                    "fine_value": by_id[source_id].fine_value,
                    "baseline_rating": int(mean_prob.argmax() + 1),
                    "baseline_expected_rating": float(np.dot(mean_prob, np.arange(1, 7))),
                    **{f"p_rating_{index + 1}": float(prob) for index, prob in enumerate(mean_prob)},
                }
            )

        preference_prompts = [
            format_chat_prompt(
                tokenizer,
                config.model.system_prompt,
                row.preference_prompt.strip() + "\n\nRespond with one concise principle.",
                max_user_tokens=config.model.max_prompt_length - 64,
            )
            for row in canonical
        ]
        pair_candidates = [[row.affirming_statement, row.opposing_statement] for row in canonical]
        pair_probabilities = scorer.score(preference_prompts, pair_candidates)
        for row, probability in zip(canonical, pair_probabilities, strict=True):
            eps = np.finfo(np.float64).tiny
            margins.append(
                {
                    "source_id": row.source_id,
                    "split": split,
                    "value": row.value,
                    "fine_value": row.fine_value,
                    "p_affirming": float(probability[0]),
                    "p_opposing": float(probability[1]),
                    "reference_affirming_margin": float(
                        np.log(probability[0] + eps) - np.log(probability[1] + eps)
                    ),
                }
            )

    output_dir = config.paths.output_root / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    rating_frame = pd.DataFrame(ratings)
    margin_frame = pd.DataFrame(margins)
    rating_frame.to_parquet(output_dir / "kvs_ratings.parquet", index=False)
    rating_frame.to_csv(output_dir / "kvs_ratings.csv", index=False)
    margin_frame.to_parquet(output_dir / "reference_margins.parquet", index=False)
    margin_frame.to_csv(output_dir / "reference_margins.csv", index=False)
    summary = {
        "model": config.model.id,
        "rating_rows": len(rating_frame),
        "margin_rows": len(margin_frame),
        "mean_expected_rating": float(rating_frame["baseline_expected_rating"].mean()),
        "reference_prefers_affirming_rate": float((margin_frame["reference_affirming_margin"] > 0).mean()),
    }
    write_json(output_dir / "summary.json", summary)
    release_model(scorer, model, tokenizer)
    return summary


def summarize_reference_mismatch(config: ProjectConfig) -> dict:
    import pandas as pd

    source = pd.read_parquet(config.paths.output_root / "baselines" / "reference_margins.parquet")
    rows = []
    for target in config.experiment.values:
        frame = source.copy()
        frame["target"] = target
        frame["is_target"] = frame["value"] == target
        frame["reference_margin"] = np.where(
            frame["is_target"], -frame["reference_affirming_margin"], frame["reference_affirming_margin"]
        )
        frame["mismatch"] = frame["reference_margin"] < 0
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    output = config.paths.output_root / "baselines" / "reference_mismatch.parquet"
    combined.to_parquet(output, index=False)
    summary_frame = combined.groupby(["target", "split", "is_target"], as_index=False).agg(
        rows=("source_id", "size"),
        mismatch_rate=("mismatch", "mean"),
        mean_margin=("reference_margin", "mean"),
    )
    summary_frame.to_csv(
        config.paths.output_root / "baselines" / "reference_mismatch_summary.csv", index=False
    )
    return {"rows": len(combined), "output": str(output)}
