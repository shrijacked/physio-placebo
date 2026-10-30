"""Subject-unit bootstrap CIs. Required for the Week 4–5 real-signal grid.

``metric`` defaults to macro-F1, the filed primary score. Balanced accuracy and
AUROC (addendum 2026-10-30) pass their own function. The second array is the
prediction for a classification metric and the score for AUROC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physio_placebo.stats.metrics import macro_f1


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float


def subject_bootstrap_ci(
    subjects: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    n: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
) -> BootstrapCI:
    """Resample subjects with replacement; ``metric`` on the pooled draw."""
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if metric is macro_f1:
        y = y.astype(int)
        yhat = yhat.astype(int)
    else:
        y = y.astype(int)
    if len(subjects) == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"))
    point = float(metric(y, yhat))
    uniq = np.unique(subjects.astype(str))
    groups = {s: np.flatnonzero(subjects.astype(str) == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[s] for s in draw])
        stats[i] = float(metric(y[idx], yhat[idx]))
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi))
