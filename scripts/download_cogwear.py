"""Download the CogWear pilot-cohort E4 CSVs file-by-file from PhysioNet.

The monolithic project zip (~188 MB) repeatedly reset mid-stream on this network and
retries were served without Range support (restarting from byte 0), so the reliable
path is the per-file HTTPS endpoint: 43 small CSVs, each verified against the
published SHA256SUMS.txt. Only pilot/*/{baseline,cognitive_load}/empatica_{bvp,eda}.csv
are fetched (the spec consumes nothing else).

Usage:
    python scripts/download_cogwear.py [--sums data/raw/cogwear/SHA256SUMS.txt]
                                       [--dest data/raw/cogwear/extracted]
                                       [--attempts 10]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/consumer-grade-wearables/1.0.0/"
WANTED_SUFFIXES = ("empatica_bvp.csv", "empatica_eda.csv")


def wanted_files(sums_path: Path) -> list[tuple[str, str]]:
    out = []
    for line in sums_path.read_text().splitlines():
        sha, _, rel = line.strip().partition(" ")
        rel = rel.strip()
        if rel.startswith("pilot/") and rel.endswith(WANTED_SUFFIXES):
            out.append((sha, rel))
    return out


def fetch(url: str, timeout_s: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sums", default="data/raw/cogwear/SHA256SUMS.txt")
    ap.add_argument("--dest", default="data/raw/cogwear/extracted")
    ap.add_argument("--attempts", type=int, default=10)
    args = ap.parse_args()

    targets = wanted_files(Path(args.sums))
    if not targets:
        raise SystemExit(f"no pilot E4 entries found in {args.sums}")
    dest = Path(args.dest)

    n_ok, failed = 0, []
    for sha, rel in targets:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha:
            n_ok += 1
            print(f"OK({n_ok}/{len(targets)}) {rel} (cached)", flush=True)
            continue
        for attempt in range(1, args.attempts + 1):
            try:
                blob = fetch(BASE + rel)
            except Exception as e:  # noqa: BLE001 - network boundary, retried
                print(f"attempt {attempt} error {rel}: {e}", flush=True)
                time.sleep(2.0)
                continue
            if hashlib.sha256(blob).hexdigest() == sha:
                out.write_bytes(blob)
                n_ok += 1
                print(f"OK({n_ok}/{len(targets)}) {rel}", flush=True)
                break
            print(f"attempt {attempt} sha mismatch {rel}", flush=True)
            time.sleep(2.0)
        else:
            failed.append(rel)

    print(f"COGWEAR_FILES_DONE ok={n_ok} of {len(targets)} failed={len(failed)}", flush=True)
    for rel in failed:
        print(f"FAILED {rel}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
