"""Prepare MAUS: read per-trial wrist PPG CSVs, cut windows, save compact arrays.

Expects the extracted MAUS distribution's raw directory, e.g.
    data/raw/maus/MAUS/Data/Raw_data
(after the user downloads MAUS.zip from IEEE DataPort — free account required.)

Usage:
    python scripts/prepare_maus.py [--raw-dir data/raw/maus/MAUS/Data/Raw_data]
                                   [--out data/processed/maus]
                                   [--config configs/windowing.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from physio_placebo.data import maus
from physio_placebo.data.windows import cut_windows, save_subject_windows
from physio_placebo.provenance import run_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/maus/MAUS/Data/Raw_data")
    ap.add_argument("--out", default="data/processed/maus")
    ap.add_argument("--config", default="configs/windowing.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["maus"]
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(
            f"MAUS raw dir not found: {raw_dir}\n"
            "Download MAUS from IEEE DataPort (DOI 10.21227/q4td-yd35, free account), "
            "extract, and point --raw-dir at .../Data/Raw_data"
        )
    out_dir = Path(args.out)
    subjects = maus.subject_ids_in_dir(raw_dir)
    if not subjects:
        raise SystemExit(f"no MAUS subjects found under {raw_dir}")

    table = {}
    for sid in tqdm(subjects, desc="subjects"):
        recs = maus.load_subject_recordings(
            raw_dir,
            sid,
            positive_nback=set(cfg["positive_nback"]),
            negative_nback=set(cfg["negative_nback"]),
        )
        sw = cut_windows("maus", sid, recs, cfg["window_s"], cfg["stride_s"])
        save_subject_windows(sw, out_dir, extra_meta={"source_raw_dir": str(raw_dir)})
        table[sid] = {
            "n_windows": sw.n_windows,
            "class_counts": {str(k): int(v)
                             for k, v in zip(*np.unique(sw.y, return_counts=True), strict=True)},
            "n_dropped_nan": sw.n_dropped_nan,
        }
        tqdm.write(f"{sid}: {table[sid]}")

    manifest = {
        **run_metadata(cfg),
        "source_raw_dir": str(raw_dir),
        "subjects": table,
        "n_subjects": len(table),
        "total_windows": sum(t["n_windows"] for t in table.values()),
    }
    (out_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(table)} subjects, {manifest['total_windows']} windows -> {out_dir}")


if __name__ == "__main__":
    main()
