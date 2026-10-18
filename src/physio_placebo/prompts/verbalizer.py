"""Paradigm B verbalizer.

Variant (i) is the original adjectives (elevated EDA → “stress-typical”).
Variant (ii) keeps the number and its percentile in the training fold, and
drops the evaluative words. Variant (iii) flips those words and keeps the
number. Thresholds and percentiles come from the training fold, never from
the person being scored. Variant (iv) is not implemented.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from physio_placebo.features.frozen import feature_columns
from physio_placebo.features.matrix import assert_matches_locked_floor, load_locked_feature_matrix

# Family priors used only in variant (i). High EDA / HR → stress-typical;
# high HRV time-domain → calmer-typical. Week 9 strips these words.
_STRESS_HIGH = {
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_phasic_std",
    "eda_scr_count",
    "eda_scr_amp_mean",
    "eda_level_mean",
    "eda_level_std",
    "hrv_mean_hr",
    "hrv_lf_hf",
}
_CALM_HIGH = {
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_mean_nn",
    "hrv_hf",
}


def train_medians(train: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        out[col] = float(np.nanmedian(train[col].to_numpy(dtype=float)))
    return out


def _percentile(value: float, train: np.ndarray) -> float | None:
    finite = np.asarray(train, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return 100.0 * float(np.mean(finite <= value))


def verbalize_row(
    row: pd.Series,
    medians: dict[str, float],
    columns: Iterable[str],
    *,
    variant: str = "i",
    train_values: dict[str, np.ndarray] | None = None,
) -> str:
    if variant not in ("i", "ii", "iii"):
        raise ValueError(f"unknown verbalizer variant {variant!r}")
    if variant == "ii" and train_values is None:
        raise ValueError("neutralized verbalizer needs the training-fold values")
    parts: list[str] = []
    for col in columns:
        val = row[col]
        if pd.isna(val):
            parts.append(f"{col} is unavailable.")
            continue
        value = float(val)
        if variant == "ii":
            pct = _percentile(value, train_values[col])
            if pct is None:
                parts.append(f"{col} is unavailable.")
            else:
                parts.append(f"{col} is {value:.4g} (percentile {pct:.0f}).")
            continue
        high = value > medians[col]
        if variant == "iii":
            high = not high
        adj = "elevated" if high else "reduced"
        if col in _STRESS_HIGH:
            typical = "stress-typical" if high else "calmer-typical"
        elif col in _CALM_HIGH:
            typical = "calmer-typical" if high else "stress-typical"
        else:
            typical = "unspecified"
        parts.append(f"{col} is {adj} ({value:.4g}), {typical}.")
    return " ".join(parts)


def verbalize_locked_dataset(dataset: str, train_subjects: Iterable[str]) -> pd.DataFrame:
    """Load the locked matrix and add a ``verbalizer_i`` column for every row."""
    assert_matches_locked_floor(dataset)
    df = load_locked_feature_matrix(dataset).copy()
    cols = feature_columns(dataset)
    train = df[df["subject"].astype(str).isin(set(map(str, train_subjects)))]
    if train.empty:
        raise ValueError(f"no training rows for subjects {list(train_subjects)}")
    medians = train_medians(train, cols)
    df["verbalizer_i"] = df.apply(lambda r: verbalize_row(r, medians, cols), axis=1)
    return df
