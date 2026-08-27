#!/usr/bin/env python3
"""Enrich KVS seeds into synthetic preference pairs with a teacher LLM.

Dry-run mode writes inspectable teacher jobs without making API calls. Normal
mode calls an OpenAI-compatible chat-completions endpoint. The teacher produces
concise public rationales, not hidden chain-of-thought traces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    parser.add_argument("--input", type=Path, default=Path("dataset/kvs_data_new.json"))
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--output", type=Path, default=Path("value_alignment/data/synthetic_preferences.jsonl"))
    parser.add_argument("--values", nargs="+", default=["all"])
    parser.add_argument("--examples-per-value", type=int, default=5)
    parser.add_argument("--personas-per-example", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--dry-run", action="store_true", help="Write teacher prompt jobs without calling an API.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between API calls.")
    return parser.parse_args()


def load_seed_examples(
    path: Path,
    split: str,
    values: list[str],
    examples_per_value: int,
    rng: random.Random,
) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in raw[split]:
        grouped[item["level2"][0]].append(item)

    selected_values = sorted(grouped) if values == ["all"] else values
    unknown = sorted(set(selected_values) - set(grouped))
    if unknown:
        raise KeyError(f"KVS values not found: {', '.join(unknown)}")

    seeds = []
    for value in selected_values:
        items = list(grouped[value])
        rng.shuffle(items)
        limit = len(items) if examples_per_value == 0 else examples_per_value
        seeds.extend(items[:limit])
    rng.shuffle(seeds)
    return seeds


def choose_personas(value: str, count: int, rng: random.Random) -> list[dict]:
    directly_relevant = [persona for persona in PERSONAS if value in persona["strong_values"]]
    remaining = [persona for persona in PERSONAS if persona not in directly_relevant]
    rng.shuffle(directly_relevant)
    rng.shuffle(remaining)
    ordered = directly_relevant + remaining
    if count <= len(ordered):
        return ordered[:count]
    return ordered + [rng.choice(PERSONAS) for _ in range(count - len(ordered))]


def build_teacher_prompt(seed: dict, persona: dict) -> str:
    value = seed["level2"][0]
    goal = seed["level1"][0]
    return f"""You are creating preference data for a DPO/HyPO value-alignment experiment.

Target value: {value}
Human-readable goal: {goal}
Persona: {persona["id"]} - {persona["description"]}

KVS positive seed:
{seed["sentence"]}

KVS contrastive seed:
{seed["negative_sentence"]}

Create one NEW, realistic everyday social or moral scenario and a response pair.

Rules:
- The prompt must describe a concrete decision and must not name the target value.
- The chosen response must clearly enact the target value and persona in that decision.
- The rejected response must be plausible and fluent but follow the contrastive orientation.
- Keep chosen and rejected similar in length and general writing quality.
- Do not copy either KVS seed sentence verbatim.
- Do not invent hidden facts or use AITA labels unless the scenario naturally asks for them.
- Keep the prompt under 180 words and each response under 120 words.
- Give only a concise, checkable public rationale; do not provide hidden chain-of-thought.

Return only valid JSON with this schema:
{{
  "prompt": "neutral decision scenario and response instruction",
  "chosen": "target-value-consistent response",
  "rejected": "contrastive response",
  "public_rationale": "one sentence explaining the observable preference"
}}"""


def call_openai_compatible(model: str, prompt: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You generate clean JSON preference data for ML experiments."},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Teacher API request failed: {exc.code} {detail}") from exc
    return json.loads(data["choices"][0]["message"]["content"])


def fallback_pair(seed: dict) -> dict[str, Any]:
    goal = seed["level1"][0].strip()
    return {
        "prompt": (
            "State one concise principle that should guide a person's decisions.\n"
            f"Goal: {goal}\n"
            "The principle should strongly support this goal. Return only the principle."
        ),
        "chosen": seed["sentence"].strip(),
        "rejected": seed["negative_sentence"].strip(),
        "public_rationale": (
            f"The chosen statement supports '{goal}', while the rejected statement is its KVS contrast."
        ),
    }


def validate_generated_pair(pair: dict) -> None:
    for field in ("prompt", "chosen", "rejected", "public_rationale"):
        if not isinstance(pair.get(field), str) or not pair[field].strip():
            raise ValueError(f"Teacher output is missing a non-empty {field!r} string.")
    if pair["chosen"].strip() == pair["rejected"].strip():
        raise ValueError("Teacher output has identical chosen and rejected responses.")


def main() -> None:
    args = parse_args()
    if args.examples_per_value < 0 or args.personas_per_example < 1:
        raise SystemExit("--examples-per-value must be non-negative and --personas-per-example must be positive.")

    rng = random.Random(args.seed)
    seeds = load_seed_examples(args.input, args.split, args.values, args.examples_per_value, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            value = seed["level2"][0]
            for persona in choose_personas(value, args.personas_per_example, rng):
                teacher_prompt = build_teacher_prompt(seed, persona)
                metadata = {
                    "source": "kvs_seed",
                    "source_split": args.split,
                    "teacher_model": args.model,
                    "target_value": value,
                    "level1": seed["level1"][0],
                    "persona_id": persona["id"],
                }
                if args.dry_run:
                    row = {
                        "metadata": metadata,
                        "teacher_prompt": teacher_prompt,
                        "fallback_pair": fallback_pair(seed),
                    }
                else:
                    generated = call_openai_compatible(args.model, teacher_prompt)
                    validate_generated_pair(generated)
                    row = {**generated, "metadata": metadata}
                    if args.sleep:
                        time.sleep(args.sleep)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1

    print(
        json.dumps(
            {
                "source": f"kvs:{args.split}",
                "seed_examples": len(seeds),
                "rows": row_count,
                "output": str(args.output),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
