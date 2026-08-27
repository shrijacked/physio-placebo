"""CogWear loader (PhysioNet 'consumer-grade-wearables' 1.0.0, pilot cohort).

Task: cognitive load. Binary label: Stroop (cognitive_load) = 1 vs resting
baseline = 0. Only the 11-subject pilot cohort (ids 0-10) is used; the
survey_gamification cohort follows a different protocol and is excluded.

Raw layout (verified against the published SHA256SUMS.txt, README.md, and
sample files on 2026-08-27):

    pilot/<id>/baseline/empatica_bvp.csv        columns 'bvp,time', 64 Hz
    pilot/<id>/baseline/empatica_eda.csv        columns 'eda,time', 4 Hz
    pilot/<id>/cognitive_load/empatica_bvp.csv
    pilot/<id>/cognitive_load/empatica_eda.csv

'time' is unix-epoch seconds. Channels within a session neither start nor stop
at identical instants, so each session's channels are trimmed to their common
overlap; after trimming, sample index round(t * fs) is valid for every channel,
which is what the windowing code assumes. Muse EEG, Samsung BVP, and E4
temperature files exist upstream but are outside the spec (E4 BVP + EDA only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .windows import Recording, Segment

SUBJECT_IDS = [str(i) for i in range(11)]
BVP_FS = 64.0
EDA_FS = 4.0
SESSIONS: tuple[tuple[str, int], ...] = (("baseline", 0), ("cognitive_load", 1))
_CHANNEL_FILES = {"bvp": "empatica_bvp.csv", "eda": "empatica_eda.csv"}
_CHANNEL_FS = {"bvp": BVP_FS, "eda": EDA_FS}


def subject_ids_in_dir(raw_dir: Path | str) -> list[str]:
    raw_dir = Path(raw_dir)
    found = []
    for sid in SUBJECT_IDS:
        needed = [
            raw_dir / "pilot" / sid / session / fname
            for session, _ in SESSIONS
            for fname in _CHANNEL_FILES.values()
        ]
        if all(p.exists() for p in needed):
            found.append(sid)
    return found


def _read_channel(path: Path, column: str, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Read one E4 CSV; validate its clock against the nominal rate."""
    df = pd.read_csv(path, usecols=[column, "time"])
    x = df[column].to_numpy(dtype=np.float64)
    t = df["time"].to_numpy(dtype=np.float64)
    if x.size < 2:
        raise ValueError(f"{path}: fewer than 2 samples")
    dt = float(np.median(np.diff(t)))
    if not np.isclose(dt, 1.0 / fs, rtol=0.01):
        raise ValueError(
            f"{path}: median sample spacing {dt:.6f}s does not match nominal "
            f"{1.0 / fs:.6f}s ({fs} Hz)"
        )
    return x, t


def load_subject_recordings(raw_dir: Path | str, subject: str) -> list[Recording]:
    """One Recording per session (baseline=0, cognitive_load=1), channels overlap-trimmed."""
    raw_dir = Path(raw_dir)
    recordings: list[Recording] = []
    for session, label in SESSIONS:
        session_dir = raw_dir / "pilot" / subject / session

        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, fname in _CHANNEL_FILES.items():
            raw[name] = _read_channel(session_dir / fname, name, _CHANNEL_FS[name])

        t_start = max(t[0] for _, t in raw.values())
        t_end = min(t[-1] for _, t in raw.values())
        if t_end <= t_start:
            raise ValueError(f"{session_dir}: channels have no temporal overlap")

        channels: dict[str, np.ndarray] = {}
        for name, (x, t) in raw.items():
            keep = (t >= t_start) & (t <= t_end)
            channels[name] = x[keep]

        duration_s = min(
            channels[name].shape[0] / _CHANNEL_FS[name] for name in channels
        )
        seg = Segment(0.0, duration_s, label, session)
        recordings.append(
            Recording(channels=channels, fs=dict(_CHANNEL_FS), segments=[seg])
        )
    return recordings
