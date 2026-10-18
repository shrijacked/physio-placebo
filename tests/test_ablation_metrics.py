"""Week 9 summaries: margin loss is gated, and the flip rate is paired by minute."""

from __future__ import annotations

import numpy as np

from physio_placebo.prompts.ablation_metrics import (
    flip_rate_report,
    neutralization_report,
    paired_flips,
    paired_wilcoxon,
)


def test_neutralization_is_undefined_when_the_original_interval_includes_chance():
    report = neutralization_report(f1_i=0.50, ci_lo_i=0.40, f1_ii=0.42, chance=0.41)
    assert report["defined"] is False
    assert report["prediction_p2"] is None
    assert report["fraction_of_margin_lost"] is None


def test_neutralization_loss_uses_the_pre_registered_ratio():
    # margin 0.20 -> 0.05 loses 75% of the margin.
    report = neutralization_report(f1_i=0.62, ci_lo_i=0.50, f1_ii=0.47, chance=0.42)
    assert report["defined"] is True
    assert report["fraction_of_margin_lost"] == (0.20 - 0.05) / 0.20
    assert report["prediction_p2"] is True


def test_flip_rate_ignores_failures_and_pairs_on_window_id():
    rows_i = [
        {"window_id": "a", "subject": "s1", "yhat": 0, "failure": False},
        {"window_id": "b", "subject": "s1", "yhat": 1, "failure": False},
        {"window_id": "c", "subject": "s2", "yhat": 1, "failure": True},
    ]
    rows_iii = [
        {"window_id": "a", "subject": "s1", "yhat": 1, "failure": False},
        {"window_id": "b", "subject": "s1", "yhat": 1, "failure": False},
        {"window_id": "c", "subject": "s2", "yhat": 0, "failure": False},
    ]
    subjects, flipped = paired_flips(rows_i, rows_iii)
    assert list(subjects) == ["s1", "s1"]
    assert list(flipped) == [True, False]
    report = flip_rate_report(subjects, flipped, n=200, seed=1337)
    assert report["rate"] == 0.5
    assert report["n_items"] == 2
    assert report["prediction_p3"] is True
    assert report["ci_lo"] <= report["rate"] <= report["ci_hi"]


def test_wilcoxon_needs_five_people():
    small = paired_wilcoxon({"a": 0.2, "b": 0.4}, {"a": 0.1, "b": 0.3})
    assert small["pvalue"] is None
    per_i = {str(i): 0.8 for i in range(6)}
    per_ii = {str(i): 0.4 for i in range(6)}
    paired = paired_wilcoxon(per_i, per_ii)
    assert paired["n"] == 6
    assert paired["pvalue"] is not None
    assert paired["pvalue"] < 0.05


def test_flip_rate_interval_is_stable_for_a_fixed_seed():
    subjects = np.array(["s1"] * 4 + ["s2"] * 4)
    flipped = np.array([True, False, True, False, False, False, False, False])
    first = flip_rate_report(subjects, flipped, n=300, seed=1337)
    second = flip_rate_report(subjects, flipped, n=300, seed=1337)
    assert first == second
