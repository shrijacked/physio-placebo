"""Adjective-vote baseline for Paradigm B. No language model.

A stress-high feature contributes +1 when it is above the training-fold median
and -1 when it is not. A calm-high feature does the opposite. Everything else,
including a missing value, contributes 0. The training fold is everyone except
the person being scored. Variant (iii) negates that vote. The prediction is B
when the vote is positive and A otherwise, including a tie. Definitions are in
``docs/analysis-addendum-2026-10-30.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from physio_placebo.prompts.verbalizer import _CALM_HIGH, _STRESS_HIGH, train_medians
from physio_placebo.surrogates.materialize import window_id


def adjective_votes(
    df: pd.DataFrame,
    train_subjects: Iterable[str],
    columns: Iterable[str],
    variant: str,
) -> np.ndarray:
    """One vote per row of ``df``, using medians from ``train_subjects`` only."""
    if variant not in ("i", "iii"):
        raise ValueError(f"adjective vote is defined for variants i and iii, not {variant!r}")
    cols = list(columns)
    train_ids = set(map(str, train_subjects))
    train = df[df["subject"].astype(str).isin(train_ids)]
    if train.empty:
        raise ValueError("adjective vote has no training rows")
    medians = train_medians(train, cols)
    votes = np.zeros(len(df), dtype=float)
    for col in cols:
        values = df[col].to_numpy(dtype=float)
        high = values > medians[col]
        if col in _STRESS_HIGH:
            contrib = np.where(high, 1.0, -1.0)
        elif col in _CALM_HIGH:
            contrib = np.where(high, -1.0, 1.0)
        else:
            continue
        votes += np.where(np.isfinite(values), contrib, 0.0)
    if variant == "iii":
        votes = -votes
    return votes


def vote_predictions(votes: np.ndarray) -> np.ndarray:
    """B (1) only when the vote is strictly positive. Ties go to A."""
    return (np.asarray(votes) > 0).astype(int)


def eval_vote_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    eval_subjects: Sequence[str],
    variant: str,
) -> pd.DataFrame:
    """Votes for evaluation subjects. Each person's median excludes that person."""
    work = df.copy()
    work["subject"] = work["subject"].astype(str)
    if "window_id" not in work.columns:
        work["window_id"] = [
            window_id(str(s), str(seg), float(t0))
            for s, seg, t0 in zip(work["subject"], work["seg_name"], work["t0"], strict=True)
        ]
    everyone = sorted(work["subject"].unique())
    held = [str(s) for s in eval_subjects]
    parts: list[pd.DataFrame] = []
    for subject in held:
        train = [s for s in everyone if s != subject]
        block = work[work["subject"] == subject].copy()
        if block.empty:
            continue
        # Medians come from the other people. The held-out rows are appended
        # so the returned tail lines up with ``block``.
        train_rows = work[work["subject"].isin(train)]
        joined = pd.concat([train_rows, block], ignore_index=True)
        votes = adjective_votes(joined, train, columns, variant)
        block["vote"] = votes[-len(block) :]
        block["yhat"] = vote_predictions(block["vote"].to_numpy())
        parts.append(block[["subject", "window_id", "y", "vote", "yhat"]])
    if not parts:
        raise ValueError("no evaluation rows for the adjective vote")
    return pd.concat(parts, ignore_index=True)


def compare_to_model(vote_frame: pd.DataFrame, rows: Sequence[dict]) -> dict:
    """Agreement with the model's letter, and Spearman of vote against its margin."""
    by_id = vote_frame.set_index("window_id")
    yhat_model: list[int] = []
    yhat_vote: list[int] = []
    margins: list[float] = []
    votes: list[float] = []
    for row in rows:
        if row.get("failure") or row.get("yhat") is None:
            continue
        if row["window_id"] not in by_id.index:
            continue
        item = by_id.loc[row["window_id"]]
        yhat_model.append(int(row["yhat"]))
        yhat_vote.append(int(item["yhat"]))
        margins.append(float(row["logprobs"]["B"]) - float(row["logprobs"]["A"]))
        votes.append(float(item["vote"]))
    if not yhat_model:
        return {"n": 0, "agreement": float("nan"), "spearman": float("nan")}
    corr = spearmanr(votes, margins).statistic
    return {
        "n": len(yhat_model),
        "agreement": float(np.mean(np.asarray(yhat_model) == np.asarray(yhat_vote))),
        "spearman": float(corr),
    }
