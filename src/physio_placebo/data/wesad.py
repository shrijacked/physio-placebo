"""WESAD loader (UCI / Uni-Siegen distribution, WESAD.zip).

Standard binary task: stress (protocol code 2) vs non-stress (baseline 1 + amusement 3).
Meditation (4) and undefined/transient codes (0, 5, 6, 7) are excluded.

Channels kept (per experimental spec §2.1): chest ECG / EDA / Resp at 700 Hz,
wrist BVP at 64 Hz, wrist EDA at 4 Hz. Other channels (ACC, EMG, Temp) are not used.

Disk discipline: the 2.25 GB zip is kept; per-subject pickles (~1 GB each) are
extracted to a temp dir one at a time and deleted immediately after windowing,
because the full unpacked dataset (~17 GB) does not fit on this machine.
"""

from __future__ import annotations

import pickle
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .windows import Recording, Segment

CHEST_FS = 700.0
WRIST_FS = {"bvp": 64.0, "eda_wrist": 4.0}
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

_MEMBER_RE = re.compile(r"^WESAD/(S\d+)/(S\d+)\.pkl$")


def subject_ids_in_zip(zip_path: Path | str) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        ids = {
            m.group(1)
            for name in z.namelist()
            if (m := _MEMBER_RE.match(name)) and m.group(1) == m.group(2)
        }
    return sorted(ids, key=lambda s: int(s[1:]))


def label_runs_to_segments(
    labels: np.ndarray,
    fs: float,
    positive: set[int],
    negative: set[int],
) -> list[Segment]:
    """Contiguous runs of protocol codes -> labeled segments (task classes only)."""
    labels = np.asarray(labels).ravel()
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [labels.shape[0]]))
    segments: list[Segment] = []
    for s, e in zip(starts, ends, strict=True):
        code = int(labels[s])
        if code in positive:
            y = 1
        elif code in negative:
            y = 0
        else:
            continue
        segments.append(Segment(s / fs, e / fs, y, LABEL_NAMES.get(code, f"code{code}")))
    return segments


def load_subject_recording(
    zip_path: Path | str,
    subject: str,
    positive: set[int] = frozenset({2}),
    negative: set[int] = frozenset({1, 3}),
) -> Recording:
    member = f"WESAD/{subject}/{subject}.pkl"
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as td:
        extracted = z.extract(member, td)
        with open(extracted, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).ravel()

    channels = {
        "ecg": np.asarray(chest["ECG"], dtype=np.float64).ravel(),
        "eda": np.asarray(chest["EDA"], dtype=np.float64).ravel(),
        "resp": np.asarray(chest["Resp"], dtype=np.float64).ravel(),
        "bvp": np.asarray(wrist["BVP"], dtype=np.float64).ravel(),
        "eda_wrist": np.asarray(wrist["EDA"], dtype=np.float64).ravel(),
    }
    fs = {"ecg": CHEST_FS, "eda": CHEST_FS, "resp": CHEST_FS, **WRIST_FS}
    segments = label_runs_to_segments(labels, CHEST_FS, set(positive), set(negative))
    return Recording(channels=channels, fs=fs, segments=segments)
