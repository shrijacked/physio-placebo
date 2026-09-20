"""Known-logprob fixture for the Week-4 scorer (no GPU, no generation)."""

from __future__ import annotations

import pytest

from physio_placebo.scoring.logprob import (
    ScoringFailure,
    argmax_label,
    load_label_spec,
    token_for_y,
    y_from_token,
)


def test_label_spec_is_a_and_b_for_all_three_datasets():
    spec = load_label_spec()
    assert spec["tokens"] == ["A", "B"]
    for name in ("wesad", "maus", "cogwear"):
        assert set(spec["datasets"][name]) == {"A", "B"}


def test_known_logprob_fixture_picks_the_larger_token():
    # Hand-set table: A is clearly more likely than B.
    logprobs = {"A": -0.10, "B": -2.30, "C": -0.01}
    assert argmax_label(logprobs, labels=("A", "B")) == "A"
    assert y_from_token(argmax_label(logprobs)) == 0


def test_known_logprob_fixture_opposite_winner():
    logprobs = {"A": -4.0, "B": -0.2}
    assert argmax_label(logprobs) == "B"
    assert y_from_token("B") == 1


def test_missing_label_is_a_scoring_failure_not_imputed():
    with pytest.raises(ScoringFailure, match="missing"):
        argmax_label({"A": -0.1}, labels=("A", "B"))


def test_tie_breaks_toward_first_token():
    assert argmax_label({"A": -1.0, "B": -1.0}) == "A"


def test_y_token_roundtrip():
    assert token_for_y(0) == "A"
    assert token_for_y(1) == "B"
    assert y_from_token(token_for_y(0)) == 0
    assert y_from_token(token_for_y(1)) == 1
