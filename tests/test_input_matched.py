"""Input-matched ceilings: a linear model on what the prompt actually contains."""

from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.baselines.classical import make_pipeline
from physio_placebo.baselines.input_matched import (
    prompt_stat_ceiling,
    prompt_vector_stats,
)
from physio_placebo.data.loso import loso_folds
from physio_placebo.paths import data_processed
from physio_placebo.stats.secondary import auroc


def test_prompt_stats_separate_classes_that_differ_in_spread():
    rng = np.random.default_rng(0)
    rows_x = []
    y = []
    subjects = []
    for subject in ("a", "b", "c"):
        for label, scale in ((0, 0.1), (1, 3.0)):
            for _ in range(30):
                rows_x.append(rng.normal(0.0, scale, size=128))
                y.append(label)
                subjects.append(subject)
    stats = np.vstack([prompt_vector_stats(x, "bin_mean") for x in rows_x])
    y_arr = np.array(y)
    sub_arr = np.array(subjects)
    pred_score = np.empty(len(y_arr))
    for test_subject, _train in loso_folds(["a", "b", "c"]):
        test = sub_arr == test_subject
        pipe = make_pipeline("logreg_l2", 1337)
        pipe.fit(stats[~test], y_arr[~test])
        pred_score[test] = pipe.decision_function(stats[test])
    assert auroc(y_arr, pred_score) > 0.9


def test_wesad_bin_mean_ceiling_matches_the_audit():
    """Full roster, the number from the 2026-10-30 audit: F1 0.786, AUROC 0.907."""
    if not any(data_processed("wesad").glob("*.npz")):
        pytest.skip("WESAD windows are not on this machine")
    ceiling = prompt_stat_ceiling("wesad", "bin_mean")
    assert ceiling["all_macro_f1"] == pytest.approx(0.786, abs=0.01)
    assert ceiling["all_auroc"] == pytest.approx(0.907, abs=0.01)
    assert ceiling["channel"] == "ecg"
    assert ceiling["n_eval"] == 906
