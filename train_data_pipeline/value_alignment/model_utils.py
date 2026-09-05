"""Shared model-loading helpers and model aliases for the value alignment project."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MODEL_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "falcon3-7b": "tiiuae/Falcon3-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def load_aliases(path: Path | None = None) -> dict[str, str]:
    aliases = dict(DEFAULT_MODEL_ALIASES)
    if path is not None and path.exists():
        aliases.update(json.loads(path.read_text(encoding="utf-8")))
    return aliases


def resolve_model_name(model: str, aliases_path: Path | None = None) -> str:
    aliases = load_aliases(aliases_path)
    return aliases.get(model, model)


def write_default_aliases(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_MODEL_ALIASES, indent=2), encoding="utf-8")
