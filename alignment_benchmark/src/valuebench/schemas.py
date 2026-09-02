"""Validated records shared by data generation, training, and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TeacherCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    affirming_statement: str = Field(min_length=12, max_length=500)
    opposing_statement: str = Field(min_length=12, max_length=500)
    preference_prompt: str = Field(min_length=12, max_length=600)
    public_rationale: str = Field(min_length=12, max_length=500)
    fidelity_confidence: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def distinct_pair(self) -> "TeacherCompletion":
        if self.affirming_statement.casefold().strip() == self.opposing_statement.casefold().strip():
            raise ValueError("Affirming and opposing statements must differ")
        return self


class CanonicalKVSRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    split: Literal["train", "eval", "test"]
    source_index: int
    category: str
    fine_value: str
    value: str
    goal: str
    source_affirming: str
    source_opposing: str
    affirming_statement: str
    opposing_statement: str
    preference_prompt: str
    public_rationale: str
    fidelity_confidence: int = Field(ge=1, le=5)
    teacher_model: str
    teacher_resolved_model: str
    teacher_prompt_version: str
    teacher_prompt_sha256: str
    teacher_request_id: str | None = None
    teacher_input_tokens: int | None = None
    teacher_output_tokens: int | None = None
    generated_at_utc: str


class AITARow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    fine_value: str
    value: str
    post: str
    high_standard_stance: Literal["NTA", "NEUTRAL", "YTA"]
    low_standard_stance: Literal["NTA", "NEUTRAL", "YTA"]

    @model_validator(mode="after")
    def distinct_stances(self) -> "AITARow":
        if self.high_standard_stance == self.low_standard_stance:
            raise ValueError("High- and low-standard stances must differ")
        return self
