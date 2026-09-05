#!/usr/bin/env python3
"""Compute CAA vectors from KVS value-affirming/opposing descriptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from value_alignment.activation_steering import (
    attention_module,
    decoder_layers,
    hidden_from_output,
)
from value_alignment.experiment_utils import write_run_manifest
from value_alignment.model_utils import resolve_model_name
from value_alignment.survey_data import make_survey_prompt
from value_alignment.value_taxonomy import basic_value_for_fine, canonical_basic_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-value", required=True)
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-descriptions", type=int, default=0)
    parser.add_argument(
        "--variant-mode",
        choices=["all", "first"],
        default="all",
        help="all uses all 27 survey prompt/template variants per description.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def contrastive_prompts(data: dict, target_value: str, variant_mode: str) -> tuple[list[str], list[str], int]:
    positives = []
    negatives = []
    descriptions = 0
    tasks = data["tasks"] if variant_mode == "all" else data["tasks"][:1]
    templates = (
        data["response_template"]
        if variant_mode == "all"
        else data["response_template"][:1]
    )
    for split in ("train", "eval"):
        for item in data[split]:
            if basic_value_for_fine(item["level2"][0]) != target_value:
                continue
            descriptions += 1
            for task in tasks:
                for template in templates:
                    positive, _ = make_survey_prompt(task, item["sentence"], template)
                    negative, _ = make_survey_prompt(task, item["negative_sentence"], template)
                    positives.append(positive)
                    negatives.append(negative)
    return positives, negatives, descriptions


class ActivationAccumulator:
    def __init__(self, model) -> None:
        self.layers, self.layer_path = decoder_layers(model)
        self.sums: dict[str, dict[str, list[torch.Tensor | None]]] = {
            site: {
                polarity: [None] * len(self.layers)
                for polarity in ("positive", "negative")
            }
            for site in ("block", "attn")
        }
        self.counts = {
            site: {polarity: [0] * len(self.layers) for polarity in ("positive", "negative")}
            for site in ("block", "attn")
        }
        self.polarity = "positive"
        self.last_positions: torch.Tensor | None = None
        self.handles = []
        for layer_index, layer in enumerate(self.layers):
            attention, _ = attention_module(layer)
            self.handles.append(
                layer.register_forward_hook(self._make_hook("block", layer_index))
            )
            self.handles.append(
                attention.register_forward_hook(self._make_hook("attn", layer_index))
            )

    def _make_hook(self, site: str, layer_index: int):
        def hook(_module, _inputs, output):
            if self.last_positions is None:
                return None
            hidden = hidden_from_output(output)
            positions = self.last_positions.to(hidden.device)
            batch_indexes = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[batch_indexes, positions].detach().float().sum(dim=0).cpu()
            current = self.sums[site][self.polarity][layer_index]
            self.sums[site][self.polarity][layer_index] = (
                selected if current is None else current + selected
            )
            self.counts[site][self.polarity][layer_index] += hidden.shape[0]
            return None

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def vectors(self) -> dict[str, torch.Tensor]:
        result = {}
        for site in ("block", "attn"):
            layer_vectors = []
            for layer_index in range(len(self.layers)):
                positive_count = self.counts[site]["positive"][layer_index]
                negative_count = self.counts[site]["negative"][layer_index]
                if positive_count == 0 or negative_count == 0:
                    raise ValueError(f"No captured activations for {site} layer {layer_index}.")
                positive = self.sums[site]["positive"][layer_index] / positive_count
                negative = self.sums[site]["negative"][layer_index] / negative_count
                layer_vectors.append((negative - positive).float())
            result[site] = torch.stack(layer_vectors)
        return result


def capture_prompts(
    model,
    tokenizer,
    accumulator: ActivationAccumulator,
    prompts: list[str],
    polarity: str,
    batch_size: int,
    max_length: int,
) -> None:
    accumulator.polarity = polarity
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(next(model.parameters()).device)
        accumulator.last_positions = encoded["attention_mask"].sum(dim=1).long() - 1
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    accumulator.last_positions = None


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_descriptions < 0:
        raise ValueError("--max-descriptions must be non-negative.")
    target = canonical_basic_value(args.target_value)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    positives, negatives, description_count = contrastive_prompts(
        data,
        target,
        args.variant_mode,
    )
    if not positives:
        raise ValueError(f"No KVS train/eval descriptions found for {target}.")
    if args.max_descriptions > 0:
        variants_per_description = len(positives) // description_count
        prompt_limit = args.max_descriptions * variants_per_description
        positives = positives[:prompt_limit]
        negatives = negatives[:prompt_limit]
        description_count = min(description_count, args.max_descriptions)
    resolved_model = resolve_model_name(args.model, args.model_aliases)
    initial_metadata = {
        "model": resolved_model,
        "target_value": target,
        "description_count": description_count,
        "contrastive_prompt_count_per_polarity": len(positives),
        "variant_mode": args.variant_mode,
        "direction": "mean(negative activation) - mean(positive activation)",
        "sites": ["block", "attn"],
    }
    write_run_manifest(
        args.output_dir,
        args,
        {"kvs_data": args.input, "model_aliases": args.model_aliases},
        metadata=initial_metadata,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires CUDA and bitsandbytes.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
    model.eval()

    accumulator = ActivationAccumulator(model)
    try:
        capture_prompts(
            model,
            tokenizer,
            accumulator,
            positives,
            "positive",
            args.batch_size,
            args.max_length,
        )
        capture_prompts(
            model,
            tokenizer,
            accumulator,
            negatives,
            "negative",
            args.batch_size,
            args.max_length,
        )
        vectors = accumulator.vectors()
    finally:
        accumulator.remove()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / "vectors.pt"
    metadata = {
        **initial_metadata,
        "layer_count": vectors["block"].shape[0],
        "hidden_size": vectors["block"].shape[1],
    }
    torch.save({**vectors, "metadata": metadata}, vector_path)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_manifest(
        args.output_dir,
        args,
        {"kvs_data": args.input, "model_aliases": args.model_aliases},
        metadata={**metadata, "vector_file": str(vector_path)},
        status="completed",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
