"""Week-2 classical floor, locked BEFORE any LLM inference runs.

Models: L2 logistic regression (C=1.0, lbfgs) and a depth-3 decision tree, both on
the identical frozen feature matrix, under strict LOSO. Preprocessing (median
imputation + z-scoring) is fit on the training fold only — no statistic ever
crosses the held-out-subject boundary.

Chance is defined two ways (pre-registered in Week 3):
1. majority-class: per fold, predict the training fold's majority class for every
   test window; pooled macro-F1.
2. empirical permutation floor: window labels shuffled independently WITHIN each
   subject (preserves per-subject class balance, destroys signal-label alignment),
   full LOSO retrain per permutation, pooled macro-F1; n_permutations draws.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..data.loso import loso_folds
from ..stats.metrics import macro_f1, per_subject_macro_f1

MODEL_NAMES = ("logreg_l2", "tree_depth3")


def make_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "logreg_l2":
        # sklearn's default penalty IS l2; naming it explicitly is deprecated in 1.9.
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    elif model_name == "tree_depth3":
        clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
    else:
        raise ValueError(f"unknown model {model_name!r}")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def loso_predict(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_name: str,
    seed: int,
) -> np.ndarray:
    """Every row predicted exactly once, by the model trained without its subject."""
    y_pred = np.full(y.shape, -1, dtype=np.int64)
    for test_subject, _train_subjects in loso_folds(list(np.unique(subjects))):
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        pipe = clone(make_pipeline(model_name, seed))
        pipe.fit(X[train_mask], y[train_mask])
        y_pred[test_mask] = pipe.predict(X[test_mask])
    assert np.all(y_pred >= 0), "some rows were never predicted"
    return y_pred


def majority_class_predictions(y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per fold, predict the training fold's majority class (ties -> lowest label)."""
    y_pred = np.full(y.shape, -1, dtype=np.int64)
    for test_subject, _ in loso_folds(list(np.unique(subjects))):
        test_mask = subjects == test_subject
        y_train = y[~test_mask]
        values, counts = np.unique(y_train, return_counts=True)
        y_pred[test_mask] = int(values[np.argmax(counts)])
    assert np.all(y_pred >= 0)
    return y_pred


def _permute_within_subject(y: np.ndarray, subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_perm = y.copy()
    for sub in np.unique(subjects):
        idx = np.flatnonzero(subjects == sub)
        y_perm[idx] = y_perm[idx][rng.permutation(idx.size)]
    return y_perm


def permutation_floor(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_name: str,
    seed: int,
    n_permutations: int,
    progress: Callable[[Iterable], Iterable] = lambda it: it,
) -> np.ndarray:
    """Null distribution of pooled macro-F1 under within-subject label permutation."""
    scores = np.empty(n_permutations, dtype=np.float64)
    for p in progress(range(n_permutations)):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(p,)))
        y_perm = _permute_within_subject(y, subjects, rng)
        y_pred = loso_predict(X, y_perm, subjects, model_name, seed)
        scores[p] = macro_f1(y_perm, y_pred)
    return scores


def run_classical_floor(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    progress: Callable[[Iterable], Iterable] = lambda it: it,
) -> dict:
    """Full Week-2 floor for one dataset's feature frame. Returns a JSON-ready dict."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.int64)
    subjects = df["subject"].to_numpy()
    seed = int(config["seed"])
    n_perm = int(config["n_permutations"])

    values, counts = np.unique(y, return_counts=True)
    out: dict = {
        "n_windows": int(y.shape[0]),
        "n_subjects": int(np.unique(subjects).shape[0]),
        "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts, strict=True)},
        "feature_columns": feature_cols,
        "nan_fraction_per_feature": {
            c: float(df[c].isna().mean()) for c in feature_cols
        },
        "models": {},
        "chance": {},
    }

    for model_name in config["models"]:
        y_pred = loso_predict(X, y, subjects, model_name, seed)
        per_sub = per_subject_macro_f1(subjects, y, y_pred)
        vals = np.array(list(per_sub.values()))
        out["models"][model_name] = {
            "pooled_macro_f1": macro_f1(y, y_pred),
            "per_subject_macro_f1": per_sub,
            "subject_mean": float(vals.mean()),
            "subject_std": float(vals.std(ddof=1)),
        }

    y_maj = majority_class_predictions(y, subjects)
    per_sub_maj = per_subject_macro_f1(subjects, y, y_maj)
    vals = np.array(list(per_sub_maj.values()))
    out["chance"]["majority_class"] = {
        "pooled_macro_f1": macro_f1(y, y_maj),
        "per_subject_macro_f1": per_sub_maj,
        "subject_mean": float(vals.mean()),
        "subject_std": float(vals.std(ddof=1)),
    }

    out["chance"]["permutation"] = {}
    for model_name in config["models"]:
        scores = permutation_floor(X, y, subjects, model_name, seed, n_perm, progress)
        out["chance"]["permutation"][model_name] = {
            "n_permutations": n_perm,
            "mean": float(scores.mean()),
            "p05": float(np.percentile(scores, 5)),
            "p50": float(np.percentile(scores, 50)),
            "p95": float(np.percentile(scores, 95)),
            "scores": [float(s) for s in scores],
        }
    return out
