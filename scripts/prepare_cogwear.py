"""Prepare CogWear: window the pilot-cohort E4 CSVs and save compact arrays.

Raw files come either from the PhysioNet zip (members matching
pilot/*/{baseline,cognitive_load}/empatica_{bvp,eda}.csv are extracted; Muse/Samsung/
temperature files and the survey_gamification cohort stay inside the zip) or from
per-file HTTPS downloads already placed under --raw-dir (each verified against the
published SHA256SUMS.txt). If the zip is absent but --raw-dir is populated, the zip
step is skipped.

Known upstream gap: pilot/3/cognitive_load/empatica_eda.csv does not exist on
PhysioNet (verified against SHA256SUMS.txt, 2026-08-27), so subject 3 fails the
complete-channels check and is excluded; the manifest records this.

Usage:
    python scripts/prepare_cogwear.py [--zip data/raw/cogwear/cogwear-1.0.0.zip]
                                      [--raw-dir data/raw/cogwear/extracted]
                                      [--out data/processed/cogwear]
                                      [--config configs/windowing.yaml]
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from physio_placebo.data import cogwear
from physio_placebo.data.windows import cut_windows, save_subject_windows
from physio_placebo.provenance import run_metadata, sha256_file

_WANTED_SUFFIXES = ("empatica_bvp.csv", "empatica_eda.csv")


def _extract_pilot_csvs(zip_path: Path, raw_dir: Path) -> int:
    """Extract needed members, stripping any top-level zip prefix before 'pilot/'."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if "pilot" not in parts or not member.endswith(_WANTED_SUFFIXES):
                continue
            rel = Path(*parts[parts.index("pilot") :])
            target = raw_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src:
                target.write_bytes(src.read())
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/cogwear/cogwear-1.0.0.zip")
    ap.add_argument("--raw-dir", default="data/raw/cogwear/extracted")
    ap.add_argument("--out", default="data/processed/cogwear")
    ap.add_argument("--config", default="configs/windowing.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["cogwear"]
    zip_path = Path(args.zip)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)

    if zip_path.exists():
        n_extracted = _extract_pilot_csvs(zip_path, raw_dir)
        print(f"extracted {n_extracted} pilot E4 CSVs -> {raw_dir}")
        source = {"source_zip": str(zip_path), "source_zip_sha256": sha256_file(zip_path)}
    elif raw_dir.exists():
        print(f"zip absent; using per-file downloads already under {raw_dir}")
        source = {"source": "per-file HTTPS downloads verified against SHA256SUMS.txt"}
    else:
        raise SystemExit(f"neither {zip_path} nor {raw_dir} exists")

    subjects = cogwear.subject_ids_in_dir(raw_dir)
    if not subjects:
        raise SystemExit(f"no CogWear pilot subjects found under {raw_dir}")
    excluded = {}
    for sid in cogwear.SUBJECT_IDS:
        if sid in subjects:
            continue
        missing = [
            f"pilot/{sid}/{session}/{fname}"
            for session, _ in cogwear.SESSIONS
            for fname in ("empatica_bvp.csv", "empatica_eda.csv")
            if not (raw_dir / "pilot" / sid / session / fname).exists()
        ]
        excluded[sid] = {"reason": "incomplete channels", "missing_files": missing}

    table = {}
    for sid in tqdm(subjects, desc="subjects"):
        recs = cogwear.load_subject_recordings(raw_dir, sid)
        sw = cut_windows("cogwear", sid, recs, cfg["window_s"], cfg["stride_s"])
        save_subject_windows(sw, out_dir, extra_meta={"source_zip": str(zip_path)})
        table[sid] = {
            "n_windows": sw.n_windows,
            "class_counts": {str(k): int(v)
                             for k, v in zip(*np.unique(sw.y, return_counts=True), strict=True)},
            "n_dropped_nan": sw.n_dropped_nan,
        }
        tqdm.write(f"{sid}: {table[sid]}")

    manifest = {
        **run_metadata(cfg),
        **source,
        "subjects": table,
        "subjects_excluded": excluded,
        "n_subjects": len(table),
        "total_windows": sum(t["n_windows"] for t in table.values()),
    }
    (out_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {len(table)} subjects, {manifest['total_windows']} windows -> {out_dir}")


if __name__ == "__main__":
    main()
