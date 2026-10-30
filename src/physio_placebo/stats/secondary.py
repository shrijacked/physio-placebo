"""Post-registration secondary metrics (addendum 2026-10-30).

These do not replace macro-F1. Balanced accuracy has chance 0.5 for any
label-independent predictor. AUROC uses the logprob margin
``logprobs["B"] - logprobs["A"]``. The item-variation gate refuses AUROC when
the margin barely moves, because identical prompts still return a few distinct
logprobs and those are ordered with the label.

The matched permutation floor shuffles labels within each subject and leaves
the predictions fixed. Holm adjusts the null-rung Wilcoxon p-values; a missing
p-value stays missing and is not part of the family.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from physio_placebo.stats.metrics import macro_f1


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of the two class hit rates. NaN if either class is absent."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    recalls: list[float] = []
    for label in (0, 1):
        mask = y_true == label
        if int(mask.sum()) == 0:
            return float("nan")
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls))


def margins(rows: Sequence[dict]) -> np.ndarray:
    """``m = logprobs["B"] - logprobs["A"]``. B is the positive class."""
    return np.array(
        [float(row["logprobs"]["B"]) - float(row["logprobs"]["A"]) for row in rows],
        dtype=float,
    )


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Pooled AUROC. NaN when either class is missing."""
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(score, dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def item_variation_ok(
    m: np.ndarray,
    deleted_std: float,
    *,
    factor: float = 10.0,
    min_noise: float = 0.01,
) -> bool:
    """True when ``std(m)`` clears ten times the deleted-cell noise floor."""
    values = np.asarray(m, dtype=float)
    noise = float(deleted_std)
    if values.size < 2 or noise != noise:
        return False
    return float(np.std(values)) > factor * max(noise, float(min_noise))


def matched_permutation_floor(
    subjects: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    n: int = 1000,
    seed: int = 1337,
) -> dict[str, float]:
    """Macro-F1 under within-subject label shuffles, predictions held fixed."""
    subjects = np.asarray(subjects).astype(str)
    labels = np.asarray(y, dtype=int)
    pred = np.asarray(yhat, dtype=int)
    groups = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]
    rng = np.random.default_rng(seed)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        shuffled = labels.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        scores[i] = macro_f1(shuffled, pred)
    return {"mean": float(scores.mean()), "p95": float(np.quantile(scores, 0.95))}


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm adjustment. None and NaN pass through and are left out of the family."""
    values = list(pvalues)
    keep = [i for i, p in enumerate(values) if p is not None and float(p) == float(p)]
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    m = len(keep)
    for rank, i in enumerate(sorted(keep, key=lambda j: float(values[j]))):
        running = max(running, (m - rank) * float(values[i]))
        adjusted[i] = min(1.0, running)
    return adjusted
