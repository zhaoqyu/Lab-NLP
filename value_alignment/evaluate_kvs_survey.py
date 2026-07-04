#!/usr/bin/env python3
"""Evaluate a model on KVS 1-6 value survey prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("HF_CACHE_DIR")
login(hf_token)

RATING_RE = re.compile(r"[1-6]")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-file", type=Path, default=Path("value_alignment/data/kvs_test_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("value_alignment/results/kvs_scores.json"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser.parse_args()


def extract_rating(text: str) -> int | None:
    match = RATING_RE.search(text)
    if match is None:
        return None
    return int(match.group(0))


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=cache_dir,
    )
    model.eval()

    rows = [json.loads(line) for line in args.eval_file.open(encoding="utf-8")]
    scored = []
    by_value: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        rating = extract_rating(generated)
        row = {**row, "generated": generated.strip(), "rating": rating}
        scored.append(row)
        if rating is not None:
            by_value[row["value"]].append(rating)

    summary = {
        value: {
            "count": len(scores),
            "mean_rating": sum(scores) / len(scores) if scores else None,
        }
        for value, scores in sorted(by_value.items())
    }
    result = {"summary": summary, "rows": scored}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
