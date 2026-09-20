"""Real-signal eval grid. Reads frozen prompt IDs. Prompt-dev subjects are never scored."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.prompts.freeze import (
    FreezeError,
    frozen_choice,
    load_frozen_prompts,
    load_frozen_prompts_c,
)
from physio_placebo.prompts.sweep import (
    _bodies_for_dataset,
    _fill_verbalizer,
    load_sweep_spec,
    score_dev_fold,
    summarize_condition,
)
from physio_placebo.scoring.client import LogprobClient
from physio_placebo.stats.bootstrap import subject_bootstrap_ci


def run_eval_cell(
    dataset: str,
    client: LogprobClient,
    *,
    model_key: str,
    paradigm: str,
    scheme: str | None,
) -> dict:
    lock = load_frozen_prompts()
    if not lock.get("frozen"):
        raise FreezeError("prompts are not frozen")
    choice = frozen_choice(dataset, model_key, paradigm, scheme)
    spec = load_sweep_spec()
    seed = int(spec["seed"])
    all_subjects = sorted(load_locked_feature_matrix(dataset)["subject"].astype(str).unique())
    dev, eval_subs = prompt_dev_split(all_subjects, n_dev=int(spec["n_dev"]), seed=seed)
    need = list(eval_subs) + list(dev)
    frame = _bodies_for_dataset(dataset, paradigm, scheme, subjects=need)
    fold_frames = []
    for held in eval_subs:
        train = [s for s in need if s != held]
        work = frame
        if paradigm == "B":
            work = _fill_verbalizer(frame, train)
        held_rows = work[work["subject"].astype(str) == held]
        pool = work[work["subject"].astype(str).isin(dev)]
        fold_frames.append(
            score_dev_fold(
                client,
                held_rows,
                dataset=dataset,
                template_id=choice["template_id"],
                n_shot=int(choice["n_shot"]),
                exemplar_pool=pool,
                seed=seed,
            )
        )
    scored = pd.concat(fold_frames, ignore_index=True)
    scored_subs = sorted(scored["subject"].astype(str).unique())
    if set(scored_subs) & set(dev):
        raise FreezeError("prompt-dev subjects leaked into eval scoring")
    summary = summarize_condition(scored)
    ok = scored[~scored["failure"].astype(bool)]
    if ok.empty:
        ci_lo = ci_hi = float("nan")
    else:
        ci = subject_bootstrap_ci(
            ok["subject"].to_numpy(),
            ok["y"].to_numpy(dtype=int),
            ok["yhat"].to_numpy(dtype=int),
            n=1000,
            seed=seed,
        )
        ci_lo, ci_hi = ci.lo, ci.hi
    return {
        "dataset": dataset,
        "model": model_key,
        "paradigm": paradigm,
        "scheme": scheme,
        "template_id": choice["template_id"],
        "n_shot": int(choice["n_shot"]),
        "prompt_dev_subjects": list(dev),
        "eval_subjects": list(eval_subs),
        "scored_subjects": scored_subs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        **summary,
    }


def run_model_eval(client: LogprobClient, model_key: str) -> list[dict]:
    lock = load_frozen_prompts()
    winners = [w for w in lock["winners"] if w["model"] == model_key]
    if not winners:
        try:
            clock = load_frozen_prompts_c()
        except FreezeError:
            clock = {"winners": []}
        winners = [w for w in clock.get("winners", []) if w["model"] == model_key]
    if not winners:
        raise FreezeError(f"no frozen cells for model {model_key}")
    rows: list[dict] = []
    for w in winners:
        rows.append(
            run_eval_cell(
                w["dataset"],
                client,
                model_key=model_key,
                paradigm=w["paradigm"],
                scheme=w["scheme"],
            )
        )
    return rows


def models_in_lock() -> Sequence[str]:
    return tuple(dict.fromkeys(w["model"] for w in load_frozen_prompts()["winners"]))
