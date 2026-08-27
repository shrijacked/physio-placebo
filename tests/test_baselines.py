from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physio_placebo.baselines.classical import (
    majority_class_predictions,
    run_classical_floor,
)

CONFIG = {"seed": 1337, "n_permutations": 20, "models": ["logreg_l2", "tree_depth3"]}
FEATURES = ["f_sep1", "f_sep2", "f_noise", "f_with_nans"]


@pytest.fixture(scope="module")
def separable_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(8):
        for i in range(30):
            y = i % 2
            mu = 1.0 if y else -1.0
            rows.append(
                {
                    "subject": f"S{s}",
                    "window_id": f"S{s}:{i}",
                    "y": y,
                    "f_sep1": rng.normal(mu, 0.3),
                    "f_sep2": rng.normal(-mu, 0.3),
                    "f_noise": rng.normal(0, 1),
                    "f_with_nans": rng.normal(mu, 0.5) if rng.random() > 0.1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_classical_floor_end_to_end(separable_frame):
    res = run_classical_floor(separable_frame, FEATURES, CONFIG)
    assert res["n_windows"] == 240 and res["n_subjects"] == 8
    lr = res["models"]["logreg_l2"]
    assert lr["pooled_macro_f1"] >= 0.9
    assert len(lr["per_subject_macro_f1"]) == 8
    tree = res["models"]["tree_depth3"]
    assert tree["pooled_macro_f1"] >= 0.8

    maj = res["chance"]["majority_class"]["pooled_macro_f1"]
    assert maj <= 0.55  # balanced classes -> constant predictor macro-F1 ~= 1/3

    for model in CONFIG["models"]:
        perm = res["chance"]["permutation"][model]
        assert perm["n_permutations"] == 20
        assert len(perm["scores"]) == 20
        assert perm["mean"] < res["models"][model]["pooled_macro_f1"]
        assert 0.1 <= perm["mean"] <= 0.7


def test_floor_is_deterministic(separable_frame):
    a = run_classical_floor(separable_frame, FEATURES, CONFIG)
    b = run_classical_floor(separable_frame, FEATURES, CONFIG)
    assert a == b


def test_majority_class_predictor_uses_train_fold_only():
    # 3 subjects; subject S0 is all-positive, the rest are mostly negative. When S0
    # is held out, the training majority is negative -> S0 predicted all 0.
    y = np.array([1] * 10 + [0] * 8 + [1] * 2 + [0] * 9 + [1] * 1)
    subjects = np.array(["S0"] * 10 + ["S1"] * 10 + ["S2"] * 10)
    y_pred = majority_class_predictions(y, subjects)
    assert np.all(y_pred[:10] == 0)


def test_nan_features_handled_by_imputer(separable_frame):
    df = separable_frame.copy()
    df.loc[df.subject == "S3", "f_with_nans"] = np.nan  # a whole subject of NaNs
    res = run_classical_floor(df, FEATURES, {**CONFIG, "n_permutations": 5})
    assert np.isfinite(res["models"]["logreg_l2"]["pooled_macro_f1"])
