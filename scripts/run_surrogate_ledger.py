"""Classical L2 logistic regression on every surrogate feature matrix.

    python scripts/run_surrogate_ledger.py
"""

from __future__ import annotations

import json

import pandas as pd

from physio_placebo.baselines.classical import loso_predict
from physio_placebo.features.frozen import DATASET_FEATURE_PLAN, feature_columns
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import run_metadata, utc_stamp
from physio_placebo.stats.metrics import macro_f1, per_subject_macro_f1
from physio_placebo.surrogates.ladder import ALL_SURROGATES, NULL_RUNGS
from physio_placebo.surrogates.materialize import SURROGATE_SEED


def _score(dataset: str, kind: str) -> dict:
    base = {
        "dataset": dataset,
        "surrogate": kind,
        "is_null": kind in NULL_RUNGS,
        "seed": SURROGATE_SEED,
    }
    if kind == "deleted":
        return {**base, "macro_f1": None, "note": "no signal; logistic regression not fit"}
    path = (
        REPO_ROOT / "results" / "surrogates" / "features" / dataset / kind / "features.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    cols = feature_columns(dataset)
    if frame[cols].notna().to_numpy().sum() == 0:
        return {**base, "macro_f1": None, "n_rows": int(len(frame)), "note": "all features NaN"}
    y = frame["y"].to_numpy(dtype=int)
    subjects = frame["subject"].to_numpy(dtype=str)
    pred = loso_predict(frame[cols].to_numpy(dtype=float), y, subjects, "logreg_l2", SURROGATE_SEED)
    return {
        **base,
        "macro_f1": macro_f1(y, pred),
        "per_subject": per_subject_macro_f1(subjects, y, pred),
        "n_rows": int(len(frame)),
        "features_parquet": str(path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    rows = []
    for dataset in DATASET_FEATURE_PLAN:
        for kind in ALL_SURROGATES:
            row = _score(dataset, kind)
            rows.append(row)
            print(f"{dataset} {kind}: {row.get('macro_f1')}")
    dest = REPO_ROOT / "results" / "surrogates" / "ledger" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=False)
    payload = {**run_metadata({"seed": SURROGATE_SEED, "model": "logreg_l2"}), "rows": rows}
    (dest / "ledger.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(dest / "ledger.json")


if __name__ == "__main__":
    main()
