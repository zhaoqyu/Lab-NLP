"""Typed project configuration with paths resolved from the YAML file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .taxonomy import BASIC_VALUES, to_basic


class PathsConfig(BaseModel):
    project_root: Path
    kvs: Path
    aita: Path
    output_root: Path
    official_hypo: Path


class TeacherConfig(BaseModel):
    model: str = "openai/gpt-5.6-sol"
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"] = "max"
    max_tokens: int = 4000
    concurrency: int = 4
    max_attempts: int = 4
    min_confidence: int = 4
    prompt_version: str = "kvs-canonical-v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_semantic_similarity: float = 0.52


class ModelConfig(BaseModel):
    id: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    max_length: int = 512
    max_prompt_length: int = 320
    trust_remote_code: bool = False
    system_prompt: str


class TrainingConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [13, 42, 97])
    target_rating: int = Field(default=1, ge=1, le=6)
    epochs: float = 3.0
    train_batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    optimizer: str = "paged_adamw_8bit"
    sft_learning_rate: float = 1e-4
    preference_learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    ipo_beta: float = 0.1
    simpo_beta: float = 2.0
    simpo_gamma: float = 1.0
    orpo_beta: float = 0.1
    early_stopping_patience: int = 2
    save_total_limit: int = 1


class EvaluationConfig(BaseModel):
    batch_size: int = 4
    kvs_prompt_variants: int = Field(default=3, ge=1, le=3)
    aita_max_per_fine_value: int = 500
    aita_labels: list[str] = Field(default_factory=lambda: ["NTA", "NEUTRAL", "YTA"])
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    min_aita_main_table_n: int = 30
    steering_sites: list[Literal["block", "attn"]] = Field(default_factory=lambda: ["block", "attn"])
    steering_layers: list[float | int] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    steering_coefficients: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0])
    steering_drift_penalty: float = 1.0


class ExperimentConfig(BaseModel):
    values: list[str] = Field(default_factory=lambda: list(BASIC_VALUES))
    trainable_methods: list[str]
    steering_methods: list[str]
    train_method_controls: bool = True

    @model_validator(mode="after")
    def canonicalize_values(self) -> "ExperimentConfig":
        self.values = list(dict.fromkeys(to_basic(value) for value in self.values))
        unknown = set(self.values) - set(BASIC_VALUES)
        if unknown:
            raise ValueError(f"Unknown values: {sorted(unknown)}")
        return self


class ProjectConfig(BaseModel):
    paths: PathsConfig
    teacher: TeacherConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig


def _absolute(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    project_root = _absolute(config_path.parent, paths["project_root"])
    paths["project_root"] = project_root
    for key in ("kvs", "aita", "output_root", "official_hypo"):
        paths[key] = _absolute(project_root, paths[key])
    if output_override := os.environ.get("VALUEBENCH_OUTPUT_ROOT"):
        paths["output_root"] = _absolute(project_root, output_override)
    config = ProjectConfig.model_validate(raw)
    if not config.paths.kvs.is_file():
        raise FileNotFoundError(config.paths.kvs)
    if not config.paths.aita.is_file():
        raise FileNotFoundError(config.paths.aita)
    return config


def config_fingerprint(config: ProjectConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
