"""Classical ceilings on the same input the language model actually sees.

Paradigms A and C show one channel, not the 18-feature matrix behind the 0.918
WESAD floor. Two ceilings, both the locked Week-2 logistic regression under
leave-one-subject-out (addendum 2026-10-30):

- three summary statistics of the Paradigm A prompt vector (mean, standard
  deviation, mean absolute difference), real windows only;
- the ``hrv_`` columns of the locked feature matrix.

The card reports the evaluation subjects. ``all_macro_f1`` / ``all_auroc`` keep
the full-roster number so it can be checked against the audit. Surrogate
windows are not scored here: regenerating them would overwrite locked files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from physio_placebo.baselines.classical import make_pipeline
from physio_placebo.data.loso import loso_folds, prompt_dev_split
from physio_placebo.data.windows import load_subject_windows
from physio_placebo.features.frozen import feature_columns
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.paradigms.downsample import apply_scheme
from physio_placebo.paradigms.series import primary_channel
from physio_placebo.paths import data_processed
from physio_placebo.stats.metrics import macro_f1
from physio_placebo.stats.secondary import auroc

PROMPT_DEV_SEED = 1337
PROMPT_DEV_N = 2


def prompt_vector_stats(x: np.ndarray, scheme: str) -> np.ndarray:
    """Mean, standard deviation, and mean absolute step of the prompt vector."""
    values = apply_scheme(np.asarray(x, dtype=float), scheme)
    if values.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    step = float(np.abs(np.diff(values)).mean()) if values.size > 1 else float("nan")
    return np.array([float(values.mean()), float(values.std()), step])


def _fit_predict(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(y.shape, -1, dtype=int)
    score = np.full(y.shape, np.nan, dtype=float)
    for test_subject, _train in loso_folds(list(np.unique(subjects))):
        test = subjects == test_subject
        pipe = make_pipeline("logreg_l2", PROMPT_DEV_SEED)
        pipe.fit(X[~test], y[~test])
        pred[test] = pipe.predict(X[test])
        score[test] = pipe.decision_function(X[test])
    if np.any(pred < 0):
        raise RuntimeError("a row was never scored")
    return pred, score


def _summarize(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    subjects: np.ndarray,
    eval_subjects: Sequence[str],
) -> dict:
    keep = np.isin(subjects, list(eval_subjects))
    return {
        "all_macro_f1": macro_f1(y, pred),
        "all_auroc": auroc(y, score),
        "eval_macro_f1": macro_f1(y[keep], pred[keep]),
        "eval_auroc": auroc(y[keep], score[keep]),
        "n_eval": int(keep.sum()),
        "n_all": int(len(y)),
    }


def eval_split(dataset: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    subjects = sorted(load_locked_feature_matrix(dataset)["subject"].astype(str).unique())
    return prompt_dev_split(subjects, n_dev=PROMPT_DEV_N, seed=PROMPT_DEV_SEED)


def hrv_feature_ceiling(dataset: str) -> dict:
    """Locked ``hrv_`` columns only. Same pipeline as the Week-2 floor."""
    frame = load_locked_feature_matrix(dataset)
    cols = [c for c in feature_columns(dataset) if c.startswith("hrv_")]
    if not cols:
        raise ValueError(f"{dataset} has no hrv_ features")
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    pred, score = _fit_predict(frame[cols].to_numpy(dtype=float), y, subjects)
    dev, eval_subjects = eval_split(dataset)
    out = _summarize(y, pred, score, subjects, eval_subjects)
    out.update({"dataset": dataset, "ceiling": "hrv", "prompt_dev_subjects": list(dev)})
    return out


def prompt_stat_matrix(dataset: str, scheme: str) -> pd.DataFrame:
    """One row per real window: the three prompt-vector statistics."""
    rows = []
    root = data_processed(dataset)
    for path in sorted(root.glob("*.npz")):
        sw = load_subject_windows(Path(path))
        channel = primary_channel(dataset)
        series = sw.channels[channel]
        for i in range(sw.n_windows):
            stats = prompt_vector_stats(series[i], scheme)
            rows.append(
                {
                    "subject": sw.subject,
                    "y": int(sw.y[i]),
                    "mean": stats[0],
                    "std": stats[1],
                    "mad": stats[2],
                }
            )
    if not rows:
        raise FileNotFoundError(f"no windows for {dataset} under {root}")
    return pd.DataFrame(rows)


def prompt_stat_ceiling(dataset: str, scheme: str) -> dict:
    frame = prompt_stat_matrix(dataset, scheme)
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    pred, score = _fit_predict(frame[["mean", "std", "mad"]].to_numpy(dtype=float), y, subjects)
    dev, eval_subjects = eval_split(dataset)
    out = _summarize(y, pred, score, subjects, eval_subjects)
    out.update(
        {
            "dataset": dataset,
            "scheme": scheme,
            "channel": primary_channel(dataset),
            "ceiling": "prompt_stats",
            "prompt_dev_subjects": list(dev),
        }
    )
    return out
