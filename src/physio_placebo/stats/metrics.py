"""Primary metric helpers: macro-F1, pooled and per subject."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_subject_macro_f1(
    subjects: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    df = pd.DataFrame({"subject": subjects, "y": y_true, "yhat": y_pred})
    return {
        str(sub): macro_f1(g["y"].to_numpy(), g["yhat"].to_numpy())
        for sub, g in df.groupby("subject", sort=True)
    }
