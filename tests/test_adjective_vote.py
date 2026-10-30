"""Adjective vote: the words alone, with no language model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physio_placebo.baselines.adjective_vote import (
    adjective_votes,
    compare_to_model,
    eval_vote_frame,
    vote_predictions,
)
from physio_placebo.features.frozen import feature_columns
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.stats.metrics import macro_f1
from physio_placebo.stats.secondary import auroc


def _frame() -> pd.DataFrame:
    # Two training people and one scored person. eda_tonic_mean is stress-high.
    # hrv_sdnn is calm-high. Medians of the training pair are 2 and 2.
    return pd.DataFrame(
        {
            "subject": ["t1", "t2", "q"],
            "eda_tonic_mean": [1.0, 3.0, 5.0],
            "hrv_sdnn": [1.0, 3.0, 0.0],
            "y": [0, 1, 1],
        }
    )


def test_hand_built_votes_and_mis_signing_negates_them():
    df = _frame()
    cols = ["eda_tonic_mean", "hrv_sdnn"]
    # Query row: EDA above the median (+1), HRV below the median (+1). Vote +2.
    original = adjective_votes(df, ["t1", "t2"], cols, "i")
    flipped = adjective_votes(df, ["t1", "t2"], cols, "iii")
    assert flipped.tolist() == pytest.approx((-original).tolist())
    assert original[-1] == pytest.approx(2.0)
    assert vote_predictions(original)[-1] == 1
    assert vote_predictions(np.array([0.0, -1.0])).tolist() == [0, 0]


def test_a_missing_value_contributes_nothing():
    df = _frame()
    df.loc[df["subject"] == "q", "eda_tonic_mean"] = np.nan
    votes = adjective_votes(df, ["t1", "t2"], ["eda_tonic_mean", "hrv_sdnn"], "i")
    # Only the calm-high feature remains: HRV 0 is below median 2, so +1.
    assert votes[-1] == pytest.approx(1.0)


def test_eval_frame_uses_everyone_except_the_scored_person():
    df = pd.DataFrame(
        {
            "subject": ["a", "b"],
            "seg_name": ["base", "base"],
            "t0": [0.0, 0.0],
            "y": [0, 1],
            "eda_tonic_mean": [0.0, 10.0],
            "hrv_sdnn": [10.0, 0.0],
        }
    )
    frame = eval_vote_frame(df, ["eda_tonic_mean", "hrv_sdnn"], ["a", "b"], "i")
    # For person a, the median is person b.
    # EDA 0 < 10 is not elevated, so the stress-high feature contributes -1.
    # HRV 10 > 0 is elevated, so the calm-high feature contributes -1.
    row_a = frame[frame["subject"] == "a"].iloc[0]
    assert row_a["vote"] == pytest.approx(-2.0)
    assert row_a["yhat"] == 0
    assert row_a["window_id"] == "a:base:0.0"


def test_wesad_original_vote_matches_the_audit():
    """Eval minutes only. The audit number is macro-F1 0.736 and AUROC 0.857."""
    import json

    from physio_placebo.paths import REPO_ROOT

    df = load_locked_feature_matrix("wesad")
    cell = json.loads(
        (REPO_ROOT / "results/verbalizer_ablation/qwen3_8b/wesad/i/cell.json").read_text()
    )["cell"]
    frame = eval_vote_frame(df, feature_columns("wesad"), cell["eval_subjects"], "i")
    by_id = frame.set_index("window_id")
    y = []
    yhat = []
    score = []
    for row in cell["rows"]:
        item = by_id.loc[row["window_id"]]
        y.append(int(row["y"]))
        yhat.append(int(item["yhat"]))
        score.append(float(item["vote"]))
    assert macro_f1(np.array(y), np.array(yhat)) == pytest.approx(0.736, abs=0.005)
    assert auroc(np.array(y), np.array(score)) == pytest.approx(0.857, abs=0.005)
    compared = compare_to_model(frame, cell["rows"])
    assert compared["n"] == 906
    assert compared["spearman"] == pytest.approx(0.84, abs=0.02)
