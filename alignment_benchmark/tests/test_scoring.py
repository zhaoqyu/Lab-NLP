from __future__ import annotations

from valuebench.scoring import _encode_candidate


class MergeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "prompt":
            return {"input_ids": [1, 2]}
        if text == "prompt answer":
            return {"input_ids": [1, 9, 3]}
        return {"input_ids": [ord(character) for character in text]}


def test_candidate_boundary_handles_token_merging():
    ids, start = _encode_candidate(MergeTokenizer(), "prompt", " answer")
    assert ids == [1, 9, 3]
    assert start == 1


class LongTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "long prompt":
            return {"input_ids": [1, 2, 3, 4, 5]}
        return {"input_ids": [1, 2, 3, 4, 5, 6]}


def test_candidate_scoring_keeps_completion_when_prompt_is_truncated():
    ids, start = _encode_candidate(LongTokenizer(), "long prompt", " answer", max_length=3)
    assert ids == [4, 5, 6]
    assert start == 2
