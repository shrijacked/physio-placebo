"""Post-registration secondary metrics from the 2026-10-30 addendum."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.stats.bootstrap import subject_bootstrap_ci
from physio_placebo.stats.metrics import macro_f1
from physio_placebo.stats.secondary import (
    auroc,
    balanced_accuracy,
    holm,
    item_variation_ok,
    margins,
    matched_permutation_floor,
)


def test_constant_predictor_has_balanced_accuracy_one_half():
    y = np.array([0] * 70 + [1] * 30)
    yhat = np.zeros(100, dtype=int)
    assert balanced_accuracy(y, yhat) == pytest.approx(0.5)
    ci = subject_bootstrap_ci(
        np.array(["s"] * 100),
        y,
        yhat,
        n=20,
        seed=1337,
        metric=balanced_accuracy,
    )
    assert ci.point == pytest.approx(0.5)
    assert ci.lo == pytest.approx(0.5)
    assert ci.hi == pytest.approx(0.5)


def test_matched_floor_of_always_a_on_a_70_30_split_is_its_own_macro_f1():
    # Ten people, seven calm and three stress each. Always answering A.
    y = np.tile(np.array([0] * 7 + [1] * 3), 10)
    yhat = np.zeros_like(y)
    subjects = np.repeat([f"s{i}" for i in range(10)], 10)
    point = macro_f1(y, yhat)
    assert point == pytest.approx(0.4117647058823529)
    floor = matched_permutation_floor(subjects, y, yhat, n=40, seed=1337)
    assert floor["mean"] == pytest.approx(point)
    assert floor["p95"] == pytest.approx(point)


def test_auroc_is_one_for_a_perfect_score_and_one_half_for_ties():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.0, 0.1, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(y, np.array([0.2, 0.2, 0.2, 0.2])) == pytest.approx(0.5)
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_margin_is_logp_b_minus_logp_a():
    rows = [{"logprobs": {"A": -0.5, "B": -0.1}}, {"logprobs": {"A": -0.2, "B": -1.2}}]
    assert margins(rows).tolist() == pytest.approx([0.4, -1.0])


def test_variation_gate_rejects_noise_sized_margins():
    # std 0.01 against deleted-cell noise 0.0126 does not clear 10x the floor.
    m = np.array([0.0, 0.02])
    assert float(np.std(m)) == pytest.approx(0.01)
    assert item_variation_ok(m, 0.0126) is False
    assert item_variation_ok(np.linspace(-1.0, 1.0, 50), 0.0126) is True


def test_holm_on_a_known_vector_and_missing_pvalues():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    missing = holm([0.01, None, 0.04])
    assert missing[1] is None
    assert missing[0] == pytest.approx(0.02)
    assert missing[2] == pytest.approx(0.04)
    assert holm([None, float("nan")]) == [None, None]
