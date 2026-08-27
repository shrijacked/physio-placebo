"""Run the Week-2 classical floor on a frozen feature matrix.

Writes results/classical_floor/<dataset>/<utcstamp>_<confighash8>/floor.json with
macro-F1 (pooled + per subject), both chance definitions, the raw permutation
scores, and full provenance including the exact feature-matrix SHA-256 consumed —
the same hash any later LLM run must record (byte-identity requirement).

Usage:
    python scripts/run_classical_floor.py --dataset wesad --features <path/to/features.parquet>
        [--config configs/classical_floor.yaml] [--n-permutations N_OVERRIDE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from physio_placebo.baselines.classical import run_classical_floor
from physio_placebo.features import frozen
from physio_placebo.provenance import config_hash, run_metadata, sha256_file, utc_stamp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(frozen.DATASET_FEATURE_PLAN))
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default="configs/classical_floor.yaml")
    ap.add_argument("--n-permutations", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.n_permutations is not None:
        cfg["n_permutations"] = args.n_permutations

    features_path = Path(args.features)
    df = pd.read_parquet(features_path)
    feat_cols = frozen.feature_columns(args.dataset)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"feature matrix lacks expected columns: {missing}")

    results = run_classical_floor(df, feat_cols, cfg, progress=lambda it: tqdm(it, desc="permutations"))

    full_cfg = {"dataset": args.dataset, **cfg}
    out_dir = Path(f"results/classical_floor/{args.dataset}") / f"{utc_stamp()}_{config_hash(full_cfg)[:8]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        **run_metadata(full_cfg),
        "features_parquet": str(features_path),
        "features_parquet_sha256": sha256_file(features_path),
        "frozen_code": frozen.frozen_code_hash(),
        "results": results,
    }
    (out_dir / "floor.json").write_text(json.dumps(payload, indent=2))

    print(f"\n=== classical floor: {args.dataset} ===")
    print(f"windows={results['n_windows']} subjects={results['n_subjects']} "
          f"classes={results['class_counts']}")
    for name, m in results["models"].items():
        print(f"{name:12s} pooled macro-F1 = {m['pooled_macro_f1']:.4f}  "
              f"per-subject mean±sd = {m['subject_mean']:.4f} ± {m['subject_std']:.4f}")
    mc = results["chance"]["majority_class"]
    print(f"{'majority':12s} pooled macro-F1 = {mc['pooled_macro_f1']:.4f}")
    for name, p in results["chance"]["permutation"].items():
        print(f"perm[{name}]  mean = {p['mean']:.4f}  p95 = {p['p95']:.4f}  (n={p['n_permutations']})")
    print(f"\nWrote {out_dir / 'floor.json'}")


if __name__ == "__main__":
    main()
