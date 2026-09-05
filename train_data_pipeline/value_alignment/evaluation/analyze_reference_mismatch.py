#!/usr/bin/env python3
"""Measure preference/reference mismatch on target-specific KVS pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from value_alignment.evaluation.reference_mismatch_metrics import summarize_margins
from value_alignment.model_utils import resolve_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Reference model or configured model alias.")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional PEFT reference adapter.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--model-aliases", type=Path, default=Path("value_alignment/configs/model_aliases.json"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def load_rows(paths: list[Path], max_samples: int = 0) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["diagnostic_source_file"] = str(path)
                rows.append(row)
                if max_samples > 0 and len(rows) >= max_samples:
                    return rows
    return rows


def _split_prompt_answer_tokens(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    """Match the prompt-boundary handling in TRL 0.9.6 DPOTrainer."""
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    prompt_only_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_start = len(prompt_only_ids)
    if prompt_only_ids != full_ids[:response_start]:
        response_start -= 1
    prompt_ids = full_ids[:response_start]
    answer_ids = full_ids[response_start:]

    if not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
        answer_ids.append(tokenizer.eos_token_id)
    return prompt_ids, answer_ids


def tokenize_preference_pair(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, tuple[list[int], int]]:
    candidate_tokens = {
        "chosen": _split_prompt_answer_tokens(tokenizer, prompt, chosen),
        "rejected": _split_prompt_answer_tokens(tokenizer, prompt, rejected),
    }
    longer_response_length = max(len(tokens[1]) for tokens in candidate_tokens.values())
    tokenized = {}
    for name, (prompt_ids, answer_ids) in candidate_tokens.items():
        if len(prompt_ids) + longer_response_length > max_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        if len(prompt_ids) + longer_response_length > max_length:
            answer_ids = answer_ids[: max_length - max_prompt_length]
        if not answer_ids:
            raise ValueError("A preference completion tokenized to an empty sequence.")
        tokenized[name] = (prompt_ids + answer_ids, len(prompt_ids))
    return tokenized


def sequence_logprob_batch(
    model,
    tokenizer,
    candidates: list[tuple[list[int], int]],
) -> list[dict[str, float | int]]:
    pad_id = tokenizer.pad_token_id
    max_width = max(len(input_ids) for input_ids, _ in candidates)
    padded_ids = []
    attention_masks = []
    for input_ids, _ in candidates:
        padding = max_width - len(input_ids)
        padded_ids.append(input_ids + [pad_id] * padding)
        attention_masks.append([1] * len(input_ids) + [0] * padding)

    device = next(model.parameters()).device
    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)

    results = []
    for row_index, (input_ids, completion_start) in enumerate(candidates):
        completion_ids = input_tensor[row_index, completion_start : len(input_ids)]
        prediction_positions = log_probs[
            row_index,
            completion_start - 1 : len(input_ids) - 1,
        ]
        token_logps = prediction_positions.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
        total = float(token_logps.sum().cpu())
        token_count = int(token_logps.numel())
        results.append(
            {
                "sequence_logprob": total,
                "mean_token_logprob": total / token_count,
                "token_count": token_count,
            }
        )
    return results


def score_rows(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_length: int,
    max_prompt_length: int,
) -> list[dict]:
    candidates = []
    for row_index, row in enumerate(rows):
        pair = tokenize_preference_pair(
            tokenizer,
            row["prompt"],
            row["chosen"],
            row["rejected"],
            max_length,
            max_prompt_length,
        )
        for candidate_name in ("chosen", "rejected"):
            candidates.append((row_index, candidate_name, pair[candidate_name]))

    scores: dict[tuple[int, str], dict] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        batch_scores = sequence_logprob_batch(
            model,
            tokenizer,
            [candidate[2] for candidate in batch],
        )
        for (row_index, candidate_name, _), score in zip(batch, batch_scores, strict=True):
            scores[(row_index, candidate_name)] = score

    scored_rows = []
    for row_index, row in enumerate(rows):
        chosen = scores[(row_index, "chosen")]
        rejected = scores[(row_index, "rejected")]
        margin = chosen["sequence_logprob"] - rejected["sequence_logprob"]
        mean_token_margin = chosen["mean_token_logprob"] - rejected["mean_token_logprob"]
        scored_rows.append(
            {
                "id": row.get("id", row_index),
                "source_split": row.get("source_split"),
                "value": row.get("value"),
                "fine_value": row.get("fine_value"),
                "target_value": row.get("target_value"),
                "is_target": bool(row.get("is_target")),
                "chosen_logprob": chosen["sequence_logprob"],
                "rejected_logprob": rejected["sequence_logprob"],
                "chosen_mean_token_logprob": chosen["mean_token_logprob"],
                "rejected_mean_token_logprob": rejected["mean_token_logprob"],
                "chosen_token_count": chosen["token_count"],
                "rejected_token_count": rejected["token_count"],
                "reference_margin": margin,
                "mean_token_reference_margin": mean_token_margin,
                "reference_mismatch": margin < 0,
                "reference_tie": margin == 0,
                "hypo_removed_pessimistic_bonus": max(0.0, -margin),
                "source_file": row["diagnostic_source_file"],
            }
        )
    return scored_rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_prompt_length >= args.max_length:
        raise ValueError("--max-prompt-length must be smaller than --max-length.")

    rows = load_rows(args.inputs, args.max_samples)
    if not rows:
        raise ValueError("No preference rows were found.")
    resolved_model = resolve_model_name(args.model, args.model_aliases)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    scored_rows = score_rows(
        model,
        tokenizer,
        rows,
        args.batch_size,
        args.max_length,
        args.max_prompt_length,
    )
    result = {
        "metric": {
            "reference_margin": "log p_ref(chosen|prompt) - log p_ref(rejected|prompt)",
            "mismatch": "reference_margin < 0",
            "hypo_rule": "replace reference_margin with max(0, reference_margin)",
        },
        "reference_model": resolved_model,
        "reference_adapter": str(args.adapter) if args.adapter is not None else None,
        "inputs": [str(path) for path in args.inputs],
        "settings": {
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "batch_size": args.batch_size,
        },
        "summary": summarize_margins(scored_rows),
        "rows": scored_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)
    print(json.dumps(result["summary"]["by_split"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
