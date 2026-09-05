#!/usr/bin/env python3
"""Select a CAA layer using only held-out KVS survey ratings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer, set_seed

from value_alignment.activation_steering import SteeringHook, decoder_layers, load_vector_file
from value_alignment.evaluate_kvs_survey import generate_run, load_model
from value_alignment.model_utils import resolve_model_name
from value_alignment.value_taxonomy import canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-value", required=True)
    parser.add_argument("--site", choices=["block", "attn"], required=True)
    parser.add_argument("--vector-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=Path("value_alignment/data/kvs_survey/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def score_layer(
    base_ratings: list[list[int | None]],
    steered_ratings: list[list[int | None]],
) -> dict:
    run_scores = []
    run_counts = []
    for run_index, (base_run, ratings) in enumerate(
        zip(base_ratings, steered_ratings, strict=True)
    ):
        drops = []
        for base_rating, conditioned_rating in zip(base_run, ratings, strict=True):
            if base_rating is None or conditioned_rating is None:
                continue
            drops.append(float(base_rating) - float(conditioned_rating))
        run_scores.append(statistics.fmean(drops) if drops else None)
        run_counts.append(len(drops))
    valid_scores = [score for score in run_scores if score is not None]
    return {
        "target_rating_drop": statistics.fmean(valid_scores) if valid_scores else None,
        "target_rating_drop_sample_std": (
            statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
        ),
        "valid_count_mean": statistics.fmean(run_counts),
        "run_scores": run_scores,
        "run_valid_counts": run_counts,
    }


def main() -> None:
    args = parse_args()
    if args.num_runs < 1 or args.batch_size < 1:
        raise ValueError("--num-runs and --batch-size must be at least 1.")
    if args.strength <= 0:
        raise ValueError("--strength must be positive for the negative-minus-positive direction.")
    target = canonical_basic_value(args.target_value)
    eval_rows = []
    with args.eval_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if canonical_basic_value(row["value"]) == target:
                eval_rows.append(row)
    if args.max_samples > 0:
        eval_rows = eval_rows[: args.max_samples]
    if not eval_rows:
        raise ValueError(f"No held-out KVS rows found for {target}.")

    resolved_model = resolve_model_name(args.model, args.model_aliases)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(resolved_model, None)
    model_layer_count = len(decoder_layers(model)[0])
    vector_payload = load_vector_file(args.vector_file)
    vector_metadata = vector_payload.get("metadata", {})
    if vector_metadata.get("model") != resolved_model:
        raise ValueError(
            f"Vector model {vector_metadata.get('model')!r} does not match "
            f"requested model {resolved_model!r}."
        )
    if vector_metadata.get("target_value") != target:
        raise ValueError(
            f"Vector target {vector_metadata.get('target_value')!r} does not match "
            f"requested target {target!r}."
        )
    vectors = vector_payload[args.site]
    if vectors.shape[0] != model_layer_count:
        raise ValueError(
            f"Vector file has {vectors.shape[0]} layers but the model has {model_layer_count}."
        )
    layers = (
        list(dict.fromkeys(args.layers))
        if args.layers is not None
        else list(range(model_layer_count))
    )
    if any(layer < 0 or layer >= model_layer_count for layer in layers):
        raise ValueError(f"--layers must fall within 0-{model_layer_count - 1}.")

    base_ratings = []
    for run_index in range(args.num_runs):
        set_seed(args.seed + run_index)
        ratings, _ = generate_run(
            model,
            tokenizer,
            eval_rows,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
        )
        base_ratings.append(ratings)

    layer_results = []
    for layer_index in layers:
        all_ratings = []
        with SteeringHook(
            model,
            args.site,
            layer_index,
            vectors[layer_index],
            args.strength,
        ):
            for run_index in range(args.num_runs):
                set_seed(args.seed + run_index)
                ratings, _ = generate_run(
                    model,
                    tokenizer,
                    eval_rows,
                    args.batch_size,
                    args.max_new_tokens,
                    args.temperature,
                )
                all_ratings.append(ratings)
        layer_results.append(
            {
                "layer": layer_index,
                **score_layer(base_ratings, all_ratings),
            }
        )

    valid_layers = [
        row for row in layer_results if row["target_rating_drop"] is not None
    ]
    if not valid_layers:
        raise ValueError("No layer produced any parseable survey ratings.")
    selected = max(valid_layers, key=lambda row: (row["target_rating_drop"], -row["layer"]))
    result = {
        "model": resolved_model,
        "target_value": target,
        "site": args.site,
        "vector_file": str(args.vector_file.resolve()),
        "vector_metadata": vector_metadata,
        "strength": args.strength,
        "selected_layer": selected["layer"],
        "selection_metric": "held-out KVS target rating drop",
        "selected_score": selected,
        "settings": {
            "num_runs": args.num_runs,
            "temperature": args.temperature,
            "seed": args.seed,
            "evaluated_rows": len(eval_rows),
            "paired_random_seeds": True,
        },
        "layers": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
