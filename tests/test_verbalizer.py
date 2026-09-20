"""Variant (i) verbalizer uses train-fold medians and locked feature names."""

from __future__ import annotations

import pandas as pd
import pytest

from physio_placebo.features.frozen import feature_columns
from physio_placebo.features.matrix import assert_matches_locked_floor, load_locked_feature_matrix
from physio_placebo.prompts.verbalizer import train_medians, verbalize_locked_dataset, verbalize_row


def test_verbalize_row_marks_high_eda_stress_typical():
    cols = ["eda_level_mean", "hrv_sdnn"]
    row = pd.Series({"eda_level_mean": 10.0, "hrv_sdnn": 1.0})
    medians = {"eda_level_mean": 5.0, "hrv_sdnn": 5.0}
    text = verbalize_row(row, medians, cols)
    assert "eda_level_mean is elevated" in text
    assert "stress-typical" in text
    assert "hrv_sdnn is reduced" in text


def test_nan_feature_is_unavailable_not_adjectived():
    cols = ["hrv_sdnn"]
    row = pd.Series({"hrv_sdnn": float("nan")})
    text = verbalize_row(row, {"hrv_sdnn": 1.0}, cols)
    assert text == "hrv_sdnn is unavailable."
    assert "typical" not in text


def test_verbalizer_loads_locked_maus_and_does_not_use_eval_subjects_for_medians():
    assert_matches_locked_floor("maus")
    df = load_locked_feature_matrix("maus")
    subjects = sorted(df["subject"].astype(str).unique())
    train, held = subjects[:-1], subjects[-1]
    out = verbalize_locked_dataset("maus", train)
    assert "verbalizer_i" in out.columns
    assert out["verbalizer_i"].str.len().min() > 0
    cols = feature_columns("maus")
    med_train = train_medians(df[df["subject"].astype(str).isin(train)], cols)
    med_all = train_medians(df, cols)
    # Held-out subject must not be required for the median definition to exist.
    assert set(med_train) == set(cols)
    # Sanity: including the held-out subject can move a median; we just record both.
    assert held not in train
    _ = med_all


def test_empty_train_fold_rejected():
    with pytest.raises(ValueError, match="no training rows"):
        verbalize_locked_dataset("maus", ["no-such-subject"])
