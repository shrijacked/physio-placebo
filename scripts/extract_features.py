"""Extract the frozen feature matrix from prepared windows.

Writes results/features/<dataset>/<utcstamp>_<confighash8>/features.parquet plus a
manifest carrying the frozen-code hash, source manifest hash, NaN report, and full
provenance. Never overwrites an existing run directory (rule 1.7).

Usage:
    python scripts/extract_features.py --dataset wesad
        [--windows-dir data/processed/wesad] [--out-root results/features/wesad]
        [--subjects S2 S3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from physio_placebo.data.windows import load_subject_windows
from physio_placebo.features import frozen
from physio_placebo.provenance import config_hash, run_metadata, sha256_file, utc_stamp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(frozen.DATASET_FEATURE_PLAN))
    ap.add_argument("--windows-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir or f"data/processed/{args.dataset}")
    out_root = Path(args.out_root or f"results/features/{args.dataset}")
    npz_files = sorted(windows_dir.glob("*.npz"))
    if args.subjects:
        npz_files = [p for p in npz_files if p.stem in set(args.subjects)]
    if not npz_files:
        raise SystemExit(f"no prepared windows found in {windows_dir}")

    code_id = frozen.frozen_code_hash()
    prep_manifest = windows_dir / "prepare_manifest.json"
    cfg = {
        "dataset": args.dataset,
        "frozen_code": code_id,
        "windows_manifest_sha256": sha256_file(prep_manifest) if prep_manifest.exists() else None,
    }
    run_dir = out_root / f"{utc_stamp()}_{config_hash(cfg)[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    frames = []
    for npz in tqdm(npz_files, desc="subjects"):
        sw = load_subject_windows(npz)
        frames.append(frozen.build_feature_frame(sw))
    df = pd.concat(frames, ignore_index=True)

    feat_cols = frozen.feature_columns(args.dataset)
    parquet_path = run_dir / "features.parquet"
    df.to_parquet(parquet_path, index=False)

    nan_report = {c: float(df[c].isna().mean()) for c in feat_cols}
    manifest = {
        **run_metadata(cfg),
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "feature_columns": feat_cols,
        "nan_fraction_per_feature": nan_report,
        "rows_all_nan": int(df[feat_cols].isna().all(axis=1).sum()),
        "features_parquet_sha256": sha256_file(parquet_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))
    print(f"\nFeature matrix: {parquet_path}")


if __name__ == "__main__":
    main()
