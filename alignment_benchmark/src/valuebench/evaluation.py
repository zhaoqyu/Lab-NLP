"""Locked KVS-test and AITA evaluation for adapters and steering vectors."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import ProjectConfig, config_fingerprint
from .data import load_aita, load_canonical
from .io import sha256_file, utc_now, write_json
from .modeling import format_chat_prompt, load_base_model, load_tokenizer, release_model
from .prompts import aita_prompt, survey_prompt
from .scoring import CandidateScorer
from .steering import load_steering
from .taxonomy import to_basic, value_slug
from .training import checkpoint_dir


def _adapter_model(config: ProjectConfig, control_path: Path, intervention_path: Path):
    from peft import PeftModel

    base = load_base_model(config)
    model = PeftModel.from_pretrained(base, str(control_path), adapter_name="control", is_trainable=False)
    model.load_adapter(str(intervention_path), adapter_name="intervention", is_trainable=False)
    model.eval()
    return model


def _kvs_distributions(
    config: ProjectConfig, scorer: CandidateScorer, tokenizer, steering=None
) -> dict[str, np.ndarray]:
    rows = load_canonical(config, "test")
    prompts = []
    owners = []
    for row in rows:
        for variant in range(config.evaluation.kvs_prompt_variants):
            prompts.append(
                format_chat_prompt(
                    tokenizer,
                    config.model.system_prompt,
                    survey_prompt(row.affirming_statement, variant),
                    max_user_tokens=config.model.max_prompt_length - 64,
                )
            )
            owners.append(row.source_id)
    candidates = [[str(rating) for rating in range(1, 7)] for _ in prompts]
    distributions = scorer.score(prompts, candidates, steering=steering)
    grouped = defaultdict(list)
    for owner, distribution in zip(owners, distributions, strict=True):
        grouped[owner].append(distribution)
    return {source_id: np.mean(values, axis=0) for source_id, values in grouped.items()}


def _aita_distributions(
    config: ProjectConfig, scorer: CandidateScorer, tokenizer, target: str, steering=None
):
    rows = load_aita(config, target)
    prompts = [
        format_chat_prompt(
            tokenizer,
            config.model.system_prompt,
            aita_prompt(row.post),
            max_user_tokens=config.model.max_prompt_length - 64,
        )
        for row in rows
    ]
    candidates = [config.evaluation.aita_labels for _ in rows]
    return rows, scorer.score(prompts, candidates, steering=steering)


def _gain(
    control: np.ndarray, intervention: np.ndarray, labels: list[str], high: str, low: str, other: float
):
    weights = np.full(len(labels), other, dtype=np.float64)
    weights[labels.index(high)] = -1.0
    weights[labels.index(low)] = 1.0
    return float(np.dot(weights, intervention - control))


def _score_pair(config: ProjectConfig, method: str, target: str, seed: int):
    tokenizer = load_tokenizer(config)
    if method.startswith("steering_"):
        site = method.removeprefix("steering_")
        model = load_base_model(config)
        scorer = CandidateScorer(
            model,
            tokenizer,
            config.evaluation.batch_size,
            max_length=config.model.max_length,
        )
        intervention = load_steering(config, model, target, site)
        control_kvs = _kvs_distributions(config, scorer, tokenizer)
        intervention_kvs = _kvs_distributions(config, scorer, tokenizer, steering=intervention)
        aita_rows, control_aita = _aita_distributions(config, scorer, tokenizer, target)
        _, intervention_aita = _aita_distributions(config, scorer, tokenizer, target, steering=intervention)
        control_source = "base_model"
        release_model(scorer, model, tokenizer)
        return control_kvs, intervention_kvs, aita_rows, control_aita, intervention_aita, control_source

    control_path = checkpoint_dir(config, method, "control", seed) / "final"
    intervention_path = checkpoint_dir(config, method, target, seed) / "final"
    if not control_path.exists() or not intervention_path.exists():
        raise FileNotFoundError(f"Missing control or intervention adapter for {method}/{target}/seed-{seed}")
    model = _adapter_model(config, control_path, intervention_path)
    scorer = CandidateScorer(
        model,
        tokenizer,
        config.evaluation.batch_size,
        max_length=config.model.max_length,
    )
    model.set_adapter("control")
    control_kvs = _kvs_distributions(config, scorer, tokenizer)
    aita_rows, control_aita = _aita_distributions(config, scorer, tokenizer, target)
    model.set_adapter("intervention")
    intervention_kvs = _kvs_distributions(config, scorer, tokenizer)
    _, intervention_aita = _aita_distributions(config, scorer, tokenizer, target)
    release_model(scorer, model, tokenizer)
    return control_kvs, intervention_kvs, aita_rows, control_aita, intervention_aita, str(control_path)


def evaluate_run(config: ProjectConfig, *, method: str, target: str, seed: int, force: bool = False) -> dict:
    import pandas as pd

    allowed = {*config.experiment.trainable_methods, *config.experiment.steering_methods}
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    if method.startswith("steering_") and seed != -1:
        raise ValueError("Activation steering is deterministic and uses registered seed -1")
    if not method.startswith("steering_") and seed not in config.training.seeds:
        raise ValueError(f"Seed {seed} is not in registered seeds {config.training.seeds}")
    target = to_basic(target)
    output = config.paths.output_root / "results" / "raw" / method / value_slug(target) / f"seed-{seed}"
    done = output / "DONE"
    if done.exists() and not force:
        return __import__("json").loads((output / "summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    control_kvs, intervention_kvs, aita_rows, control_aita, intervention_aita, control_source = _score_pair(
        config, method, target, seed
    )

    ratings = np.arange(1, 7)
    kvs_records = []
    for row in load_canonical(config, "test"):
        control = control_kvs[row.source_id]
        intervention = intervention_kvs[row.source_id]
        control_expected = float(np.dot(control, ratings))
        intervention_expected = float(np.dot(intervention, ratings))
        kvs_records.append(
            {
                "source_id": row.source_id,
                "method": method,
                "target": target,
                "seed": seed,
                "value": row.value,
                "fine_value": row.fine_value,
                "is_target": row.value == target,
                "control_expected_rating": control_expected,
                "intervention_expected_rating": intervention_expected,
                "rating_drop": control_expected - intervention_expected,
                "absolute_drift": abs(control_expected - intervention_expected),
                **{f"control_p_{rating}": float(control[rating - 1]) for rating in range(1, 7)},
                **{f"intervention_p_{rating}": float(intervention[rating - 1]) for rating in range(1, 7)},
            }
        )
    kvs_frame = pd.DataFrame(kvs_records)
    kvs_frame.to_parquet(output / "kvs.parquet", index=False)

    aita_records = []
    labels = config.evaluation.aita_labels
    for row, control, intervention in zip(aita_rows, control_aita, intervention_aita, strict=True):
        aita_records.append(
            {
                "source_id": row.source_id,
                "method": method,
                "target": target,
                "seed": seed,
                "value": row.value,
                "fine_value": row.fine_value,
                "high_standard_stance": row.high_standard_stance,
                "low_standard_stance": row.low_standard_stance,
                "probability_gain": _gain(
                    control,
                    intervention,
                    labels,
                    row.high_standard_stance,
                    row.low_standard_stance,
                    0.5,
                ),
                "strict_probability_gain": _gain(
                    control,
                    intervention,
                    labels,
                    row.high_standard_stance,
                    row.low_standard_stance,
                    0.0,
                ),
                **{f"control_p_{label.lower()}": float(control[index]) for index, label in enumerate(labels)},
                **{
                    f"intervention_p_{label.lower()}": float(intervention[index])
                    for index, label in enumerate(labels)
                },
            }
        )
    aita_frame = pd.DataFrame(aita_records)
    aita_frame.to_parquet(output / "aita.parquet", index=False)
    summary = {
        "method": method,
        "target": target,
        "seed": seed,
        "base_model": config.model.id,
        "config_sha256": config_fingerprint(config),
        "control_source": control_source,
        "evaluated_at_utc": utc_now(),
        "kvs_rows": len(kvs_frame),
        "aita_rows": len(aita_frame),
        "target_rating_drop": float(kvs_frame.loc[kvs_frame["is_target"], "rating_drop"].mean()),
        "non_target_drift": float(kvs_frame.loc[~kvs_frame["is_target"], "absolute_drift"].mean()),
        "aita_probability_gain": float(aita_frame["probability_gain"].mean()),
        "aita_strict_probability_gain": float(aita_frame["strict_probability_gain"].mean()),
        "kvs_sha256": sha256_file(output / "kvs.parquet"),
        "aita_sha256": sha256_file(output / "aita.parquet"),
    }
    write_json(output / "summary.json", summary)
    done.write_text("completed\n", encoding="utf-8")
    return summary
