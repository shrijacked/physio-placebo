"""vLLM step parsing without importing vLLM."""

from __future__ import annotations

import pytest

from physio_placebo.scoring.logprob import ScoringFailure
from physio_placebo.scoring.vllm_extract import extract_label_logprobs


class _LP:
    def __init__(self, logprob: float):
        self.logprob = logprob


def test_extracts_named_label_logprobs_from_token_ids():
    step = {10: _LP(-0.2), 11: _LP(-1.8), 99: _LP(-0.01)}
    table = extract_label_logprobs(step, {"A": 10, "B": 11})
    assert table == {"A": -0.2, "B": -1.8}


def test_plain_float_entries_work():
    table = extract_label_logprobs({1: -0.5, 2: -3.0}, {"A": 1, "B": 2})
    assert table["A"] == -0.5


def test_missing_label_id_is_scoring_failure():
    with pytest.raises(ScoringFailure, match="missing"):
        extract_label_logprobs({10: _LP(-0.1)}, {"A": 10, "B": 11})
