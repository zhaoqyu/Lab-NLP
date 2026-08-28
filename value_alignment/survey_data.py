"""Utilities shared by KVS survey data preparation and evaluation."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator

from value_alignment.value_taxonomy import basic_value_for_fine


RATING_RE = re.compile(r"(?<!\d)([1-6])(?!\d)")


def kvs_item_id(split: str, index: int) -> str:
    return f"kvs-{split}-{index:04d}"


def extract_rating(text: str) -> int | None:
    match = RATING_RE.search(text)
    return int(match.group(1)) if match else None


def majority_rating(votes: list[int]) -> int:
    if not votes:
        raise ValueError("Cannot assign a majority rating without valid votes.")
    counts = Counter(votes)
    highest_count = max(counts.values())
    tied = [rating for rating, count in counts.items() if count == highest_count]
    if len(tied) == 1:
        return tied[0]
    mean_vote = sum(votes) / len(votes)
    return min(tied, key=lambda rating: (abs(rating - mean_vote), rating))


def split_response_template(template: str) -> tuple[str, str]:
    if template.count("{rating}") != 1:
        raise ValueError(f"Survey response template must contain one {{rating}} placeholder: {template!r}")
    return tuple(template.split("{rating}"))  # type: ignore[return-value]


def make_survey_prompt(task: str, sentence: str, response_template: str) -> tuple[str, str]:
    """Return the model prompt and suffix around the rating completion."""
    response_prefix, response_suffix = split_response_template(response_template)
    prompt = f"{task.strip()}\n\nStatement:\n{sentence.strip()}\n\n{response_prefix}"
    return prompt, response_suffix


def iter_survey_variants(
    item: dict,
    split: str,
    index: int,
    tasks: list[str],
    response_templates: list[str],
) -> Iterator[dict]:
    fine_value = item["level2"][0]
    basic_value = basic_value_for_fine(fine_value)
    for task_index, task in enumerate(tasks):
        for template_index, response_template in enumerate(response_templates):
            prompt, response_suffix = make_survey_prompt(task, item["sentence"], response_template)
            yield {
                "id": f"{kvs_item_id(split, index)}-t{task_index:02d}-r{template_index:02d}",
                "source_id": kvs_item_id(split, index),
                "source_split": split,
                "prompt": prompt,
                "response_suffix": response_suffix,
                "sentence": item["sentence"].strip(),
                "negative_sentence": item["negative_sentence"].strip(),
                "fine_value": fine_value,
                "value": basic_value,
                "category": item["category"],
                "level1": item["level1"][0],
                "task_index": task_index,
                "template_index": template_index,
            }
