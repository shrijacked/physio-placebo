"""Subject-unit bootstrap CIs. Required for the Week 4–5 real-signal grid.

SDS / Wilcoxon stay Weeks 7–10.
"""

from __future__ import annotations

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
) -> BootstrapCI:
    """Resample subjects with replacement; pooled macro-F1 each draw."""
    subjects = np.asarray(subjects)
    y = np.asarray(y, dtype=int)
    yhat = np.asarray(yhat, dtype=int)
    if len(subjects) == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"))
    point = macro_f1(y, yhat)
    uniq = np.unique(subjects.astype(str))
    groups = {s: np.flatnonzero(subjects.astype(str) == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[s] for s in draw])
        stats[i] = macro_f1(y[idx], yhat[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi))
