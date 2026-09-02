"""Full-sequence candidate scoring shared by KVS and AITA evaluations."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Protocol

import numpy as np


class SteeringLike(Protocol):
    def activate(self, start_positions: list[int]): ...


def _encode_candidate(
    tokenizer,
    prompt: str,
    candidate: str,
    max_length: int | None = None,
) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
    start = min(len(prompt_ids), len(full_ids))
    while start > 0 and full_ids[:start] != prompt_ids[:start]:
        start -= 1
    if start == len(full_ids):
        raise ValueError(f"Candidate produced no completion tokens: {candidate!r}")
    if max_length is not None and len(full_ids) > max_length:
        removed = len(full_ids) - max_length
        full_ids = full_ids[removed:]
        start -= removed
        if start < 1:
            raise ValueError("Candidate or prompt budget leaves no context token for likelihood scoring")
    return full_ids, start


class CandidateScorer:
    def __init__(self, model, tokenizer, batch_size: int = 4, max_length: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()

    def score(
        self,
        prompts: list[str],
        candidates: list[list[str]],
        *,
        steering: SteeringLike | None = None,
    ) -> list[np.ndarray]:
        """Return normalized candidate probabilities for every prompt."""
        if len(prompts) != len(candidates):
            raise ValueError("prompts and candidates must have equal length")
        flat = []
        owners = []
        for owner, (prompt, options) in enumerate(zip(prompts, candidates, strict=True)):
            if len(options) < 2:
                raise ValueError("Every prompt needs at least two candidates")
            for option in options:
                ids, start = _encode_candidate(
                    self.tokenizer,
                    prompt,
                    option,
                    max_length=self.max_length,
                )
                flat.append((ids, start))
                owners.append(owner)

        logps = []
        for offset in range(0, len(flat), self.batch_size):
            logps.extend(self._score_batch(flat[offset : offset + self.batch_size], steering=steering))

        grouped: list[list[float]] = [[] for _ in prompts]
        for owner, logp in zip(owners, logps, strict=True):
            grouped[owner].append(logp)
        probabilities = []
        for values in grouped:
            array = np.asarray(values, dtype=np.float64)
            array -= array.max()
            exp = np.exp(array)
            probabilities.append(exp / exp.sum())
        return probabilities

    def _score_batch(self, rows: list[tuple[list[int], int]], steering: SteeringLike | None) -> list[float]:
        import torch
        import torch.nn.functional as F

        pad = self.tokenizer.pad_token_id
        max_length = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), max_length), pad, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        completion_mask = torch.zeros((len(rows), max_length - 1), dtype=torch.bool)
        starts = []
        for index, (ids, start) in enumerate(rows):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention_mask[index, : len(ids)] = 1
            completion_mask[index, max(start - 1, 0) : len(ids) - 1] = True
            starts.append(max(start - 1, 0))
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)
        context = steering.activate(starts) if steering is not None else nullcontext()
        with torch.inference_mode(), context:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        token_logps = (
            F.log_softmax(shift_logits.float(), dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        )
        sequence_logps = (token_logps * completion_mask).sum(dim=-1)
        return sequence_logps.detach().cpu().tolist()
