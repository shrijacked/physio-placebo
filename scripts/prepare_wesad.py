"""Prepare WESAD: stream subjects out of the zip, cut windows, save compact arrays.

Usage:
    python scripts/prepare_wesad.py [--zip data/raw/wesad/WESAD.zip]
                                    [--out data/processed/wesad]
                                    [--config configs/windowing.yaml]
                                    [--subjects S2 S3 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from physio_placebo.data import wesad
from physio_placebo.data.windows import cut_windows, save_subject_windows
from physio_placebo.provenance import run_metadata, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/wesad/WESAD.zip")
    ap.add_argument("--out", default="data/processed/wesad")
    ap.add_argument("--config", default="configs/windowing.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["wesad"]
    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    subjects = args.subjects or wesad.subject_ids_in_zip(zip_path)

    print(f"WESAD zip: {zip_path} — hashing source archive...")
    zip_sha = sha256_file(zip_path)
    table = {}
    for sid in tqdm(subjects, desc="subjects"):
        rec = wesad.load_subject_recording(
            zip_path, sid, positive=set(cfg["positive"]), negative=set(cfg["negative"])
        )
        sw = cut_windows("wesad", sid, [rec], cfg["window_s"], cfg["stride_s"])
        save_subject_windows(sw, out_dir, extra_meta={"source_zip_sha256": zip_sha})
        table[sid] = {
            "n_windows": sw.n_windows,
            "class_counts": {str(k): int(v)
                             for k, v in zip(*np.unique(sw.y, return_counts=True), strict=True)},
            "n_dropped_nan": sw.n_dropped_nan,
        }
        tqdm.write(f"{sid}: {table[sid]}")

    manifest = {
        **run_metadata(cfg),
        "source_zip": str(zip_path),
        "source_zip_sha256": zip_sha,
        "subjects": table,
        "n_subjects": len(table),
        "total_windows": sum(t["n_windows"] for t in table.values()),
    }
    (out_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(table)} subjects, {manifest['total_windows']} windows -> {out_dir}")


if __name__ == "__main__":
    main()
