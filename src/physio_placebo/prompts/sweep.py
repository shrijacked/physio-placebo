"""Prompt-dev sweep: pick a template × shot per condition. No freeze here."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import yaml

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.frozen import feature_columns
from physio_placebo.features.matrix import assert_matches_locked_floor, load_locked_feature_matrix
from physio_placebo.paradigms.series import load_windows, series_body
from physio_placebo.paths import configs_dir
from physio_placebo.prompts.templates import TEMPLATE_IDS, render_prompt
from physio_placebo.prompts.verbalizer import train_medians, verbalize_row
from physio_placebo.scoring.client import LogprobClient
from physio_placebo.scoring.logprob import ScoringFailure, argmax_label, token_for_y, y_from_token
from physio_placebo.stats.metrics import macro_f1, per_subject_macro_f1


def load_sweep_spec() -> dict:
    return yaml.safe_load((configs_dir() / "prompt_sweep.yaml").read_text())


def pick_exemplars(
    pool: pd.DataFrame,
    n: int,
    seed: int,
) -> list[tuple[str, str]]:
    """n (body, token) pairs from other prompt-dev rows, class-balanced when possible."""
    if pool.empty:
        raise ValueError("exemplar pool is empty")
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for y in (0, 1):
        g = pool[pool["y"].astype(int) == y]
        if g.empty:
            continue
        take = min(n // 2, len(g)) if n >= 2 else min(1, len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=len(g) < take)
        parts.append(g.loc[idx])
    chosen = pd.concat(parts) if parts else pool
    if len(chosen) < n:
        extra = rng.choice(pool.index.to_numpy(), size=n - len(chosen), replace=True)
        chosen = pd.concat([chosen, pool.loc[extra]])
    chosen = chosen.iloc[:n]
    return [
        (str(r["body"]), token_for_y(int(r["y"])))
        for r in chosen.to_dict(orient="records")
    ]


def _bodies_for_dataset(
    dataset: str,
    paradigm: str,
    scheme: str | None,
    subjects: Sequence[str] | None = None,
) -> pd.DataFrame:
    assert_matches_locked_floor(dataset)
    feat = load_locked_feature_matrix(dataset)
    if subjects is not None:
        keep = set(map(str, subjects))
        feat = feat[feat["subject"].astype(str).isin(keep)].copy()
    if paradigm == "B":
        cols = feature_columns(dataset)
        # Medians recomputed per scored subject; placeholder column filled later.
        out = feat.copy()
        out["body"] = ""
        out.attrs["verbalizer_cols"] = cols
        return out
    if paradigm != "A" or scheme is None:
        raise ValueError(f"bad paradigm/scheme {paradigm}/{scheme}")
    bodies: list[str] = []
    cache: dict[str, object] = {}
    for r in feat.itertuples(index=False):
        sub = str(r.subject)
        if sub not in cache:
            cache[sub] = load_windows(dataset, sub)
        bodies.append(
            series_body(cache[sub], t0=float(r.t0), seg_name=str(r.seg_name), scheme=scheme, dataset=dataset)
        )
    out = feat.copy()
    out["body"] = bodies
    return out


def _fill_verbalizer(df: pd.DataFrame, train_subjects: Sequence[str]) -> pd.DataFrame:
    cols = df.attrs["verbalizer_cols"]
    train = df[df["subject"].astype(str).isin(set(map(str, train_subjects)))]
    medians = train_medians(train, cols)
    out = df.copy()
    out["body"] = [verbalize_row(row, medians, cols) for _, row in df.iterrows()]
    return out


def score_dev_fold(
    client: LogprobClient,
    rows: pd.DataFrame,
    *,
    dataset: str,
    template_id: str,
    n_shot: int,
    exemplar_pool: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    spec = load_sweep_spec()
    exemplars = (
        pick_exemplars(exemplar_pool, int(spec["n_exemplars"]), seed) if n_shot else None
    )
    prompts = [
        render_prompt(template_id, dataset, str(r["body"]), exemplars=exemplars)
        for r in rows.to_dict(orient="records")
    ]
    tables = client.score_prompts(prompts)
    recs = []
    for r, table, prompt in zip(rows.to_dict(orient="records"), tables, prompts, strict=True):
        try:
            tok = argmax_label(table)
            yhat = y_from_token(tok)
            fail = False
        except ScoringFailure:
            tok, yhat, fail = None, None, True
        recs.append(
            {
                "dataset": dataset,
                "subject": str(r["subject"]),
                "window_id": r["window_id"],
                "y": int(r["y"]),
                "yhat": yhat,
                "token": tok,
                "failure": fail,
                "template_id": template_id,
                "n_shot": n_shot,
                "prompt": prompt,
                "logprobs": table,
            }
        )
    return pd.DataFrame(recs)


def summarize_condition(scored: pd.DataFrame) -> dict:
    ok = scored[~scored["failure"].astype(bool)]
    if ok.empty:
        return {"macro_f1": float("nan"), "n_scored": 0, "n_failures": int(len(scored))}
    y = ok["y"].to_numpy(dtype=int)
    yhat = ok["yhat"].to_numpy(dtype=int)
    return {
        "macro_f1": macro_f1(y, yhat),
        "per_subject": per_subject_macro_f1(ok["subject"].to_numpy(), y, yhat),
        "n_scored": int(len(ok)),
        "n_failures": int(scored["failure"].sum()),
    }


def run_dataset_sweep(
    dataset: str,
    client: LogprobClient,
    *,
    paradigm: str,
    scheme: str | None,
    model_key: str,
) -> list[dict]:
    spec = load_sweep_spec()
    seed = int(spec["seed"])
    all_subjects = sorted(load_locked_feature_matrix(dataset)["subject"].astype(str).unique())
    dev, _eval = prompt_dev_split(all_subjects, n_dev=int(spec["n_dev"]), seed=seed)
    frame = _bodies_for_dataset(dataset, paradigm, scheme, subjects=dev)
    results: list[dict] = []
    for template_id in spec["template_ids"]:
        if template_id not in TEMPLATE_IDS:
            raise ValueError(template_id)
        for n_shot in spec["n_shot"]:
            fold_frames = []
            for held in dev:
                others = [s for s in dev if s != held]
                work = frame
                if paradigm == "B":
                    work = _fill_verbalizer(frame, others)
                held_rows = work[work["subject"].astype(str) == held]
                pool = work[work["subject"].astype(str).isin(others)]
                fold_frames.append(
                    score_dev_fold(
                        client,
                        held_rows,
                        dataset=dataset,
                        template_id=template_id,
                        n_shot=int(n_shot),
                        exemplar_pool=pool,
                        seed=seed,
                    )
                )
            scored = pd.concat(fold_frames, ignore_index=True)
            summary = summarize_condition(scored)
            results.append(
                {
                    "dataset": dataset,
                    "model": model_key,
                    "paradigm": paradigm,
                    "scheme": scheme,
                    "template_id": template_id,
                    "n_shot": int(n_shot),
                    "prompt_dev_subjects": list(dev),
                    **summary,
                }
            )
    return results


def pick_winner(rows: Sequence[dict]) -> dict:
    """Highest prompt-dev macro-F1; ties → 0-shot, then template list order."""
    def key(r: dict) -> tuple:
        f1 = r["macro_f1"]
        f1s = -1.0 if f1 != f1 else float(f1)  # NaN loses
        return (f1s, 0 if r["n_shot"] == 0 else -1, -TEMPLATE_IDS.index(r["template_id"]))

    return max(rows, key=key)
