"""Prompt-dev sweep helpers. ConstantClient only — no GPU, no freeze."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.paradigms.series import format_series, load_windows, window_index
from physio_placebo.prompts.sweep import pick_exemplars, pick_winner, run_dataset_sweep, summarize_condition
from physio_placebo.scoring.client import ConstantClient


def test_pick_exemplars_balanced_and_uses_tokens():
    pool = pd.DataFrame(
        {
            "body": [f"b{i}" for i in range(8)],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    ex = pick_exemplars(pool, n=4, seed=1337)
    assert len(ex) == 4
    tokens = {t for _, t in ex}
    assert tokens == {"A", "B"}


def test_pick_winner_prefers_higher_f1_then_zero_shot():
    rows = [
        {"template_id": "clinical", "n_shot": 4, "macro_f1": 0.40},
        {"template_id": "instruction", "n_shot": 0, "macro_f1": 0.40},
        {"template_id": "minimal", "n_shot": 0, "macro_f1": 0.10},
    ]
    w = pick_winner(rows)
    assert w["template_id"] == "instruction"
    assert w["n_shot"] == 0


def test_summarize_condition_counts_failures():
    scored = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2"],
            "y": [0, 1, 0],
            "yhat": [0, 1, None],
            "failure": [False, False, True],
        }
    )
    s = summarize_condition(scored)
    assert s["n_scored"] == 2
    assert s["n_failures"] == 1
    assert s["macro_f1"] == pytest.approx(1.0)


def test_format_series_includes_scheme_and_length():
    text = format_series(np.arange(1000, dtype=float), channel="ecg", fs=700.0, scheme="uniform_stride")
    assert text.startswith("ecg fs=700 Hz n=256 scheme=uniform_stride:")
    assert " 0 " in f" {text} " or text.endswith("999") or "999" in text


def test_wesad_window_index_matches_feature_row():
    df = load_locked_feature_matrix("wesad")
    row = df.iloc[0]
    sw = load_windows("wesad", str(row.subject))
    idx = window_index(sw, float(row.t0), str(row.seg_name))
    assert int(sw.y[idx]) == int(row.y)


def test_paradigm_a_sweep_loads_only_prompt_dev_windows(monkeypatch):
    """Prompt-dev sweep must not touch eval-subject npz (server only has n_dev=2)."""
    loaded: list[str] = []
    real = load_windows

    def spy(dataset: str, subject: str):
        loaded.append(str(subject))
        return real(dataset, subject)

    monkeypatch.setattr("physio_placebo.prompts.sweep.load_windows", spy)
    rows = run_dataset_sweep(
        "wesad",
        ConstantClient(),
        paradigm="A",
        scheme="uniform_stride",
        model_key="constant",
    )
    subjects = load_locked_feature_matrix("wesad")["subject"].astype(str).unique()
    dev, eval_ = prompt_dev_split(sorted(subjects), n_dev=2, seed=1337)
    assert set(loaded) == set(dev)
    assert set(eval_).isdisjoint(loaded)
    assert set(rows[0]["prompt_dev_subjects"]) == set(dev)


def test_maus_paradigm_b_sweep_runs_on_constant_client():
    rows = run_dataset_sweep(
        "maus",
        ConstantClient(),
        paradigm="B",
        scheme=None,
        model_key="constant",
    )
    assert len(rows) == 6  # 3 templates × 2 shots
    subjects = load_locked_feature_matrix("maus")["subject"].astype(str).unique()
    dev, eval_ = prompt_dev_split(subjects, n_dev=2, seed=1337)
    assert set(rows[0]["prompt_dev_subjects"]) == set(dev)
    assert set(eval_).isdisjoint(dev)
    for r in rows:
        assert r["n_scored"] > 0
        assert r["n_failures"] == 0
        assert 0.0 <= r["macro_f1"] <= 1.0
