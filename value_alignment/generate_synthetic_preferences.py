#!/usr/bin/env python3
"""Generate synthetic AITA preference pairs with a teacher LLM.

The script supports two modes:

1. `--dry-run`: write teacher prompts as JSONL jobs for inspection or later batch
   execution.
2. normal mode: call an OpenAI-compatible chat-completions endpoint using
   `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.

The requested "CoT" is stored as a short public rationale. We avoid hidden
chain-of-thought traces and ask the teacher for concise, checkable reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_VALUES = [
    "Security_personal",
    "Benevolence_caring",
    "Universalism_concern",
]

PERSONAS = [
    {
        "id": "safety_first_caregiver",
        "description": "A cautious caregiver who prioritizes health, safety, and preventing avoidable harm.",
        "strong_values": ["Security_personal", "Benevolence_caring"],
    },
    {
        "id": "fairness_mediator",
        "description": "A balanced mediator who emphasizes fairness, impartiality, and respectful compromise.",
        "strong_values": ["Universalism_concern", "Universalism_objectivity"],
    },
    {
        "id": "loyal_friend",
        "description": "A dependable friend who strongly values loyalty, reliability, and care for close others.",
        "strong_values": ["Benevolence_dependability", "Benevolence_caring"],
    },
    {
        "id": "autonomy_advocate",
        "description": "An autonomy-focused person who values self-direction, boundaries, and personal agency.",
        "strong_values": ["Self_direction_action", "Self_direction_thought"],
    },
    {
        "id": "rule_respecting_citizen",
        "description": "A rule-oriented person who values social order, clear norms, and predictable conduct.",
        "strong_values": ["Conformity_rules", "Security_societal", "Tradition"],
    },
    {
        "id": "achievement_oriented_professional",
        "description": "A goal-driven person who values competence, responsibility, and successful outcomes.",
        "strong_values": ["Achievement", "Benevolence_dependability"],
    },
    {
        "id": "environmental_universalist",
        "description": "A broad-concern person who values nature, equality, tolerance, and long-term collective welfare.",
        "strong_values": ["Universalism_nature", "Universalism_concern", "Universalism_tolerance"],
    },
    {
        "id": "dignity_and_face_preserver",
        "description": "A socially careful person who values dignity, reputation, and avoiding unnecessary humiliation.",
        "strong_values": ["Face", "Conformity_interpersonal"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/aita_dataset_reduced.json"))
    parser.add_argument("--output", type=Path, default=Path("value_alignment/data/synthetic_preferences.jsonl"))
    parser.add_argument("--values", nargs="+", default=DEFAULT_VALUES)
    parser.add_argument("--examples-per-value", type=int, default=20)
    parser.add_argument("--personas-per-example", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--dry-run", action="store_true", help="Write teacher prompt jobs without calling an API.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between API calls.")
    return parser.parse_args()


def load_seed_examples(path: Path, values: list[str], examples_per_value: int, rng: random.Random) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selected_values = list(raw) if values == ["all"] else values
    seeds: list[dict] = []
    for value in selected_values:
        if value not in raw:
            raise KeyError(f"Value {value!r} not found in {path}")
        items = list(raw[value])
        rng.shuffle(items)
        for item in items[:examples_per_value]:
            seeds.append(item)
    rng.shuffle(seeds)
    return seeds


def choose_personas(value: str, count: int, rng: random.Random) -> list[dict]:
    directly_relevant = [persona for persona in PERSONAS if value in persona["strong_values"]]
    remaining = [persona for persona in PERSONAS if persona not in directly_relevant]
    ordered = directly_relevant + remaining
    if len(ordered) >= count:
        return ordered[:count]
    return [rng.choice(PERSONAS) for _ in range(count)]


def build_teacher_prompt(seed: dict, persona: dict) -> str:
    value = seed["value"]
    high_label = seed["high_standard_stance"]
    low_label = seed["low_standard_stance"]
    high_rationale = seed["justification"]["high_standard_stance"]
    low_rationale = seed["justification"]["low_standard_stance"]

    return f"""You are creating training data for a value-alignment DPO/HyPO experiment.

Target value: {value}
Persona: {persona["id"]} - {persona["description"]}

AITA post:
{seed["post"]}

Seed labels:
- High-standard value stance: {high_label}
- Low-standard value stance: {low_label}

Seed rationales:
- High-standard rationale: {high_rationale}
- Low-standard rationale: {low_rationale}

Create one preference pair for training.

Rules:
- The chosen answer must use the high-standard value stance label.
- The rejected answer must use the low-standard value stance label.
- Each answer must start with exactly one label: NTA, YTA, or Neutral.
- Use concise public reasoning, not long hidden chain-of-thought.
- Do not invent new facts beyond the post.
- Keep each answer under 120 words.

Return only valid JSON with this schema:
{{
  "prompt": "AITA prompt for the student model",
  "chosen": "label plus concise value-grounded rationale",
  "rejected": "label plus concise lower-value rationale",
  "public_rationale": "one-sentence explanation of why the chosen answer better represents the target value"
}}"""


def call_openai_compatible(model: str, prompt: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You generate clean JSON training data for ML experiments."},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Teacher API request failed: {exc.code} {detail}") from exc

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def make_seed_prompt(seed: dict, persona: dict) -> str:
    return (
        "You are evaluating an AITA post.\n"
        f"Target value: {seed['value']}.\n"
        f"Persona context: {persona['description']}\n"
        "Give exactly one label from NTA, YTA, or Neutral, followed by a short explanation.\n\n"
        f"Post:\n{seed['post']}"
    )


def fallback_pair(seed: dict, persona: dict) -> dict[str, Any]:
    return {
        "prompt": make_seed_prompt(seed, persona),
        "chosen": f"{seed['high_standard_stance']}. {seed['justification']['high_standard_stance']}",
        "rejected": f"{seed['low_standard_stance']}. {seed['justification']['low_standard_stance']}",
        "public_rationale": seed["justification"]["value_assignment"],
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    seeds = load_seed_examples(args.input, args.values, args.examples_per_value, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            for persona in choose_personas(seed["value"], args.personas_per_example, rng):
                teacher_prompt = build_teacher_prompt(seed, persona)
                metadata = {
                    "source": "aita_seed",
                    "teacher_model": args.model,
                    "target_value": seed["value"],
                    "persona_id": persona["id"],
                    "high_standard_stance": seed["high_standard_stance"],
                    "low_standard_stance": seed["low_standard_stance"],
                }
                if args.dry_run:
                    row = {
                        "metadata": metadata,
                        "teacher_prompt": teacher_prompt,
                        "fallback_pair": fallback_pair(seed, persona),
                    }
                else:
                    generated = call_openai_compatible(args.model, teacher_prompt)
                    row = {**generated, "metadata": metadata}
                    if args.sleep:
                        time.sleep(args.sleep)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"seed_examples": len(seeds), "rows": len(seeds) * args.personas_per_example, "output": str(args.output), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
