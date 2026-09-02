"""Contrastive Activation Addition with validation-only layer/strength selection."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from .config import ProjectConfig, config_fingerprint
from .data import load_canonical
from .io import utc_now, write_json
from .modeling import (
    decoder_layers,
    format_chat_prompt,
    hardware_info,
    load_base_model,
    load_tokenizer,
    release_model,
    resolve_layer_indices,
)
from .prompts import steering_context, survey_prompt
from .scoring import CandidateScorer
from .taxonomy import to_basic, value_slug


def _module(model, layer_index: int, site: str):
    layer = decoder_layers(model)[layer_index]
    if site == "block":
        return layer
    if site == "attn" and hasattr(layer, "self_attn"):
        return layer.self_attn
    raise TypeError(f"Model layer does not expose site {site!r}")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class SteeringIntervention:
    def __init__(self, model, layer_index: int, site: str, vector, coefficient: float):
        self.model = model
        self.layer_index = layer_index
        self.site = site
        self.vector = vector
        self.coefficient = coefficient

    @contextmanager
    def activate(self, start_positions: list[int]):
        module = _module(self.model, self.layer_index, self.site)

        def hook(_module, _inputs, output):
            hidden = _hidden(output)
            if hidden.shape[0] != len(start_positions):
                raise ValueError("Steering batch size does not match start positions")
            modified = hidden.clone()
            vector = self.vector.to(device=modified.device, dtype=modified.dtype)
            for row, start in enumerate(start_positions):
                modified[row, start:, :] += self.coefficient * vector
            return _replace_hidden(output, modified)

        handle = module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def _capture_final_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_index: int,
    site: str,
    batch_size: int,
    max_length: int,
):
    import torch

    captured = []
    module = _module(model, layer_index, site)
    current_mask = None

    def hook(_module, _inputs, output):
        hidden = _hidden(output)
        positions = current_mask.sum(dim=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        captured.append(hidden[rows, positions].detach().float().cpu())

    handle = module.register_forward_hook(hook)
    try:
        for offset in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            current_mask = batch["attention_mask"]
            with torch.inference_mode():
                model(**batch, use_cache=False)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


def compute_vector(config: ProjectConfig, model, tokenizer, target: str, layer_index: int, site: str):
    import torch

    target = to_basic(target)
    rows = [row for row in load_canonical(config, "train") if row.value == target]
    positive = [
        format_chat_prompt(
            tokenizer,
            config.model.system_prompt,
            steering_context(row.affirming_statement),
            max_user_tokens=config.model.max_prompt_length - 64,
        )
        for row in rows
    ]
    negative = [
        format_chat_prompt(
            tokenizer,
            config.model.system_prompt,
            steering_context(row.opposing_statement),
            max_user_tokens=config.model.max_prompt_length - 64,
        )
        for row in rows
    ]
    pos = _capture_final_activations(
        model,
        tokenizer,
        positive,
        layer_index,
        site,
        config.evaluation.batch_size,
        config.model.max_length,
    )
    neg = _capture_final_activations(
        model,
        tokenizer,
        negative,
        layer_index,
        site,
        config.evaluation.batch_size,
        config.model.max_length,
    )
    differences = neg - pos
    refined_means = []
    for fine_value in sorted({row.fine_value for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.fine_value == fine_value]
        refined_means.append(differences[indices].mean(dim=0))
    return torch.stack(refined_means).mean(dim=0)


def _selection_examples(config: ProjectConfig, tokenizer):
    rows = load_canonical(config, "eval")
    prompts = [
        format_chat_prompt(
            tokenizer,
            config.model.system_prompt,
            survey_prompt(row.affirming_statement, 0),
            max_user_tokens=config.model.max_prompt_length - 64,
        )
        for row in rows
    ]
    candidates = [[str(rating) for rating in range(1, 7)] for _ in rows]
    return rows, prompts, candidates


def _expected(distributions: list[np.ndarray]) -> np.ndarray:
    ratings = np.arange(1, 7)
    return np.asarray([np.dot(distribution, ratings) for distribution in distributions])


def build_steering(config: ProjectConfig, *, target: str, site: str, force: bool = False) -> dict:
    import pandas as pd
    import torch

    target = to_basic(target)
    if site not in config.evaluation.steering_sites:
        raise ValueError(f"site must be one of {config.evaluation.steering_sites}")
    output = config.paths.output_root / "steering" / site / value_slug(target)
    done = output / "DONE"
    if done.exists() and not force:
        return __import__("json").loads((output / "selected.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config)
    model = load_base_model(config)
    scorer = CandidateScorer(
        model,
        tokenizer,
        config.evaluation.batch_size,
        max_length=config.model.max_length,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows, prompts, candidates = _selection_examples(config, tokenizer)
    control_expected = _expected(scorer.score(prompts, candidates))
    layer_indices = resolve_layer_indices(model, config.evaluation.steering_layers)
    records = []
    vectors = {}
    for layer_index in layer_indices:
        vector = compute_vector(config, model, tokenizer, target, layer_index, site)
        vectors[layer_index] = vector
        torch.save(vector, output / f"vector_layer_{layer_index}.pt")
        for coefficient in config.evaluation.steering_coefficients:
            intervention = SteeringIntervention(model, layer_index, site, vector, coefficient)
            intervention_expected = _expected(scorer.score(prompts, candidates, steering=intervention))
            changes = control_expected - intervention_expected
            target_drop = float(
                np.mean(
                    [
                        changes[np.asarray([row.fine_value == fine_value for row in rows])].mean()
                        for fine_value in sorted({row.fine_value for row in rows if row.value == target})
                    ]
                )
            )
            non_target_drift = float(
                np.mean(
                    [
                        np.abs(changes[np.asarray([row.value == value for row in rows])]).mean()
                        for value in sorted({row.value for row in rows if row.value != target})
                    ]
                )
            )
            objective = target_drop - config.evaluation.steering_drift_penalty * non_target_drift
            records.append(
                {
                    "target": target,
                    "site": site,
                    "layer": layer_index,
                    "coefficient": coefficient,
                    "target_drop": target_drop,
                    "non_target_drift": non_target_drift,
                    "selection_objective": objective,
                    "vector_norm": float(vector.norm()),
                }
            )
    frame = pd.DataFrame(records).sort_values(
        ["selection_objective", "target_drop", "non_target_drift"], ascending=[False, False, True]
    )
    frame.to_csv(output / "selection_grid.csv", index=False)
    best = frame.iloc[0].to_dict()
    selected_vector = vectors[int(best["layer"])]
    torch.save(selected_vector, output / "selected_vector.pt")
    selected = {
        **best,
        "selection_split": "eval",
        "vector_construction_split": "train",
        "selected_at_utc": utc_now(),
        "base_model": config.model.id,
        "config_sha256": config_fingerprint(config),
        "vector_path": str(output / "selected_vector.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "hardware": hardware_info(),
    }
    write_json(output / "selected.json", selected)
    done.write_text("completed\n", encoding="utf-8")
    release_model(scorer, model, tokenizer)
    return selected


def load_steering(config: ProjectConfig, model, target: str, site: str) -> SteeringIntervention:
    import json
    import torch

    output = config.paths.output_root / "steering" / site / value_slug(target)
    selected = json.loads((output / "selected.json").read_text(encoding="utf-8"))
    vector = torch.load(output / "selected_vector.pt", map_location="cpu", weights_only=True)
    return SteeringIntervention(
        model,
        layer_index=int(selected["layer"]),
        site=site,
        vector=vector,
        coefficient=float(selected["coefficient"]),
    )
