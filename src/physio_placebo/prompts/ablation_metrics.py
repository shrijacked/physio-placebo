"""Week 9 summaries. Chance is the locked permutation-floor mean.

The neutralization claim is defined only when variant (i) clears that floor
with a bootstrap interval that stays above it. The flip rate is the share of
scored minutes whose predicted letter changes between (i) and (iii).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def chance_mean(floor: dict) -> float:
    return float(floor["results"]["chance"]["permutation"]["logreg_l2"]["mean"])


def neutralization_report(
    f1_i: float,
    ci_lo_i: float,
    f1_ii: float,
    chance: float,
) -> dict:
    margin_i = float(f1_i) - float(chance)
    margin_ii = float(f1_ii) - float(chance)
    defined = float(ci_lo_i) > float(chance)
    loss = (margin_i - margin_ii) / margin_i if defined else None
    return {
        "defined": defined,
        "chance": float(chance),
        "margin_i": margin_i,
        "margin_ii": margin_ii,
        "fraction_of_margin_lost": loss,
        "prediction_p2": None if loss is None else bool(loss > 0.5),
    }


def paired_flips(rows_i: list[dict], rows_iii: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_i = {
        r["window_id"]: r
        for r in rows_i
        if not r.get("failure") and r.get("yhat") is not None
    }
    subjects: list[str] = []
    flipped: list[bool] = []
    for row in rows_iii:
        if row.get("failure") or row.get("yhat") is None:
            continue
        original = by_i.get(row["window_id"])
        if original is None:
            continue
        subjects.append(str(row["subject"]))
        flipped.append(int(original["yhat"]) != int(row["yhat"]))
    return np.asarray(subjects), np.asarray(flipped, dtype=bool)


def flip_rate_report(
    subjects: np.ndarray,
    flipped: np.ndarray,
    *,
    n: int = 1000,
    seed: int = 1337,
) -> dict:
    subjects = np.asarray(subjects).astype(str)
    flipped = np.asarray(flipped, dtype=bool)
    if flipped.size == 0:
        return {
            "rate": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_items": 0,
            "per_subject": {},
            "prediction_p3": None,
        }
    point = float(flipped.mean())
    uniq = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in uniq}
    per_subject = {s: float(flipped[idx].mean()) for s, idx in groups.items()}
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for i in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[s] for s in chosen])
        draws[i] = float(flipped[idx].mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "rate": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_items": int(flipped.size),
        "per_subject": per_subject,
        "prediction_p3": bool(point > 0.4),
    }


def paired_wilcoxon(per_i: dict[str, float], per_other: dict[str, float]) -> dict:
    keys = sorted(set(per_i) & set(per_other))
    if len(keys) < 5:
        return {"n": len(keys), "statistic": None, "pvalue": None}
    diff = np.array([float(per_i[k]) - float(per_other[k]) for k in keys])
    if np.all(diff == 0):
        return {"n": len(keys), "statistic": None, "pvalue": None}
    result = wilcoxon(diff)
    return {
        "n": len(keys),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }
