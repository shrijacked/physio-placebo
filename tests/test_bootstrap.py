"""Subject-unit bootstrap CIs for the Week 4–5 real-signal grid."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.stats.bootstrap import subject_bootstrap_ci
from physio_placebo.stats.metrics import macro_f1


def test_subject_bootstrap_ci_contains_point_and_is_deterministic():
    subjects = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([0, 1, 0, 1, 0, 1])
    yhat = np.array([0, 1, 0, 1, 0, 0])
    point = macro_f1(y, yhat)
    a = subject_bootstrap_ci(subjects, y, yhat, n=200, seed=1337)
    b = subject_bootstrap_ci(subjects, y, yhat, n=200, seed=1337)
    assert a == b
    assert a.lo <= point <= a.hi
    assert a.point == pytest.approx(point)


def test_subject_is_the_resample_unit():
    subjects = np.array(["only"] * 8)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    yhat = y.copy()
    ci = subject_bootstrap_ci(subjects, y, yhat, n=50, seed=1337)
    assert ci.lo == pytest.approx(ci.hi)
