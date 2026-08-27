"""MAUS loader (IEEE DataPort, DOI 10.21227/q4td-yd35).

Task: mental workload under n-back. Binary label: 0-back = low (0) vs
2-back and 3-back = high (1). The separate resting recordings are not part of
the task and are excluded.

Raw layout (verified against the authors' released baseline code,
github.com/rickwu11/MAUS_dataset_baseline_system, `src/util/read_file.py`):

    Data/Raw_data/<subject>/pixart.csv     wrist PPG, columns 'Trial 1:0back' ... 'Trial 6:0back'
    Data/Raw_data/<subject>/inf_ecg.csv    chest ECG 256 Hz (not used: spec keeps wrist PPG only)
    Data/Raw_data/<subject>/inf_ppg.csv    fingertip PPG 256 Hz (not used)

Trial-to-condition mapping (from the column headers and the baseline code):
trials 1 and 6 = 0-back, trials 2 and 4 = 2-back, trials 3 and 5 = 3-back.

PixArt wrist PPG sampling rate: 100 Hz for the first five subjects
(002, 003, 004, 005, 006) and 102.5 Hz for the rest — hard-coded the same way in
the authors' baseline (`peak_detection.py`: `if sub > 4: pix_fs = 102.5`).

Each trial is loaded as its own Recording with a single full-length segment;
trailing NaN padding is trimmed, interior NaNs cause window drops (counted).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .windows import Recording, Segment

SUBJECT_IDS = [
    "002", "003", "004", "005", "006", "008", "010", "011", "012", "013", "014",
    "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025",
]
_PIX_100HZ = {"002", "003", "004", "005", "006"}
TRIAL_COLUMNS = [
    "Trial 1:0back", "Trial 2:2back", "Trial 3:3back",
    "Trial 4:2back", "Trial 5:3back", "Trial 6:0back",
]
TRIAL_NBACK = {1: 0, 2: 2, 3: 3, 4: 2, 5: 3, 6: 0}


def pixart_fs(subject: str) -> float:
    return 100.0 if subject in _PIX_100HZ else 102.5


def subject_ids_in_dir(raw_dir: Path | str) -> list[str]:
    raw_dir = Path(raw_dir)
    return [s for s in SUBJECT_IDS if (raw_dir / s / "pixart.csv").exists()]


def _trim_nan_edges(x: np.ndarray) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return x[:0]
    return x[finite[0] : finite[-1] + 1]


def load_subject_recordings(
    raw_dir: Path | str,
    subject: str,
    positive_nback: set[int] = frozenset({2, 3}),
    negative_nback: set[int] = frozenset({0}),
) -> list[Recording]:
    fs = pixart_fs(subject)
    df = pd.read_csv(Path(raw_dir) / subject / "pixart.csv", usecols=TRIAL_COLUMNS)
    recordings: list[Recording] = []
    for i, col in enumerate(TRIAL_COLUMNS, start=1):
        nback = TRIAL_NBACK[i]
        if nback in positive_nback:
            y = 1
        elif nback in negative_nback:
            y = 0
        else:
            continue
        x = _trim_nan_edges(df[col].to_numpy(dtype=np.float64))
        if x.size == 0:
            continue
        seg = Segment(0.0, x.shape[0] / fs, y, f"trial_{i}_{nback}back")
        recordings.append(Recording(channels={"ppg": x}, fs={"ppg": fs}, segments=[seg]))
    return recordings
