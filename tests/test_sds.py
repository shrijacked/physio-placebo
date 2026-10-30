"""Filed SDS: the ratio, the gate, and the paired subject-bootstrap delta."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.stats.sds import (
    gate_defined,
    paired_subject_bootstrap_delta,
    sds,
    sds_or_undefined,
)


def test_sds_is_the_filed_ratio():
    # (0.60 - 0.41) / (0.60 - 0.42) = 0.19 / 0.18
    assert sds(0.60, 0.41, 0.42) == pytest.approx(0.19 / 0.18)


def test_negative_sds_is_returned_as_computed():
    assert sds(0.60, 0.70, 0.42) == pytest.approx((0.60 - 0.70) / (0.60 - 0.42))
    assert sds(0.60, 0.70, 0.42) < 0


def test_gate_requires_the_interval_to_sit_strictly_above_chance():
    assert gate_defined(0.50, 0.42) is True
    assert gate_defined(0.42, 0.42) is False
    assert gate_defined(0.40, 0.42) is False
    assert gate_defined(float("nan"), 0.42) is False


def test_undefined_sds_is_none_when_the_gate_fails():
    assert sds_or_undefined(0.45, 0.41, 0.42, ci_lo_real=0.40) is None
    assert sds_or_undefined(0.60, 0.41, 0.42, ci_lo_real=0.50) == pytest.approx(0.19 / 0.18)


def test_identical_conditions_have_a_zero_delta_interval():
    subjects = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([0, 1, 0, 1, 0, 1])
    yhat = np.array([0, 1, 0, 0, 1, 1])
    delta = paired_subject_bootstrap_delta(subjects, y, yhat, yhat, n=50, seed=1337)
    assert delta.point == pytest.approx(0.0)
    assert delta.lo == pytest.approx(0.0)
    assert delta.hi == pytest.approx(0.0)


def test_paired_delta_uses_the_same_subject_draw_for_both_conditions():
    subjects = np.array(["a", "a", "b", "b"])
    y = np.array([0, 1, 0, 1])
    real = np.array([0, 1, 0, 1])
    surr = np.array([1, 0, 1, 0])
    delta = paired_subject_bootstrap_delta(subjects, y, real, surr, n=20, seed=1)
    assert delta.point == pytest.approx(1.0)
    assert delta.lo == pytest.approx(1.0)
    assert delta.hi == pytest.approx(1.0)
