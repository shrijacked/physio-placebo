"""Write surrogate windows and feature matrices. Does not score models.

    python scripts/prepare_surrogates.py --dataset wesad
    python scripts/prepare_surrogates.py
"""

from __future__ import annotations

import argparse
import json

from physio_placebo.features.frozen import DATASET_FEATURE_PLAN, feature_columns
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import run_metadata, sha256_file, utc_stamp
from physio_placebo.surrogates.ladder import ALL_SURROGATES, NULL_RUNGS
from physio_placebo.surrogates.materialize import (
    SURROGATE_SEED,
    features_from_windows,
    permuted_features,
    write_signal_surrogates,
)


def _windows_dir(dataset: str, kind: str):
    return REPO_ROOT / "results" / "surrogates" / "windows" / dataset / kind


def _features_path(dataset: str, kind: str):
    return REPO_ROOT / "results" / "surrogates" / "features" / dataset / kind / "features.parquet"


def prepare_one(dataset: str, kind: str) -> dict:
    feat_path = _features_path(dataset, kind)
    if kind == "deleted":
        note = {
            "dataset": dataset,
            "surrogate": kind,
            "is_null": True,
            "features": None,
            "note": "signal deleted; no feature matrix",
        }
        dest = feat_path.parent / "none.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(note, indent=2) + "\n")
        print(f"{dataset} {kind}: no signal")
        return note
    wdir = _windows_dir(dataset, kind)
    if not any(wdir.glob("*.npz")):
        subjects = write_signal_surrogates(dataset, kind, wdir, seed=SURROGATE_SEED)
        print(f"{dataset} {kind}: wrote {len(subjects)} subjects")
    if kind.startswith("subject_perm"):
        frame = permuted_features(
            dataset,
            preserve_label=kind == "subject_perm_preserved",
            seed=SURROGATE_SEED,
        )
    else:
        frame = features_from_windows(dataset, wdir)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(feat_path, index=False)
    cols = feature_columns(dataset)
    nan_rate = float(frame[cols].isna().to_numpy().mean())
    manifest = {
        "dataset": dataset,
        "surrogate": kind,
        "is_null": kind in NULL_RUNGS,
        "n_rows": int(len(frame)),
        "feature_nan_rate": nan_rate,
        "features_parquet": str(feat_path.relative_to(REPO_ROOT)),
        "features_sha256": sha256_file(feat_path),
        "seed": SURROGATE_SEED,
        **run_metadata({"dataset": dataset, "surrogate": kind, "seed": SURROGATE_SEED}),
    }
    (feat_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{dataset} {kind}: rows={len(frame)} nan_rate={nan_rate:.3f}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_FEATURE_PLAN), default=None)
    parser.add_argument("--kind", choices=ALL_SURROGATES, default=None)
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else list(DATASET_FEATURE_PLAN)
    kinds = [args.kind] if args.kind else list(ALL_SURROGATES)
    stamp = utc_stamp()
    for dataset in datasets:
        for kind in kinds:
            prepare_one(dataset, kind)
    print(f"done {stamp}")


if __name__ == "__main__":
    main()
