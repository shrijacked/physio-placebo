"""Filed Surrogate Dependence Score and its validity gate.

SDS = (M_real - M_surrogate) / (M_real - M_chance), with M the pooled
leave-one-subject-out macro-F1 and M_chance the locked permutation-floor mean.
The gate is the same rule as the Week 9 neutralization report: the real-signal
bootstrap interval must sit strictly above chance. A ratio is never returned
when that fails. A ratio outside [0, 1] is returned as computed.

The paired delta resamples subjects once and scores both conditions on that
same draw. Rows must already be aligned on window id.
"""

from __future__ import annotations

import numpy as np

from physio_placebo.stats.bootstrap import BootstrapCI
from physio_placebo.stats.metrics import macro_f1


def sds(m_real: float, m_surr: float, m_chance: float) -> float:
    """Raw filed ratio. Raises if the denominator is zero."""
    denom = float(m_real) - float(m_chance)
    if denom == 0.0:
        raise ZeroDivisionError("SDS denominator is zero; apply the validity gate first")
    return (float(m_real) - float(m_surr)) / denom


def gate_defined(ci_lo_real: float, m_chance: float) -> bool:
    """True only when the real-signal interval sits strictly above chance."""
    lo = float(ci_lo_real)
    if lo != lo:  # NaN
        return False
    return lo > float(m_chance)


def sds_or_undefined(
    m_real: float,
    m_surr: float,
    m_chance: float,
    ci_lo_real: float,
) -> float | None:
    """Filed SDS, or None when the validity gate fails."""
    if not gate_defined(ci_lo_real, m_chance):
        return None
    return sds(m_real, m_surr, m_chance)


def paired_subject_bootstrap_delta(
    subjects: np.ndarray,
    y: np.ndarray,
    yhat_real: np.ndarray,
    yhat_surr: np.ndarray,
    *,
    n: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Pooled macro-F1(real) minus pooled macro-F1(surrogate), same subject draw."""
    subjects = np.asarray(subjects)
    y = np.asarray(y, dtype=int)
    yhat_real = np.asarray(yhat_real, dtype=int)
    yhat_surr = np.asarray(yhat_surr, dtype=int)
    if not (len(subjects) == len(y) == len(yhat_real) == len(yhat_surr)):
        raise ValueError("subjects, y, and both predictions must be aligned and the same length")
    if len(subjects) == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"))
    point = macro_f1(y, yhat_real) - macro_f1(y, yhat_surr)
    uniq = np.unique(subjects.astype(str))
    groups = {s: np.flatnonzero(subjects.astype(str) == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[s] for s in draw])
        stats[i] = macro_f1(y[idx], yhat_real[idx]) - macro_f1(y[idx], yhat_surr[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(point=float(point), lo=float(lo), hi=float(hi))
