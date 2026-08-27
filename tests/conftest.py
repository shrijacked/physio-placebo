"""Shared fixtures: synthetic physiological signals and fake dataset distributions.

The fake WESAD zip and fake MAUS raw dir replicate the real on-disk formats
(verified against the actual distributions) so loader tests exercise the real
code paths end-to-end, just on small synthetic recordings.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

CHEST_FS = 700
WESAD_PROTOCOL = [
    # (label_code, duration_s)
    (0, 20.0),
    (1, 130.0),  # baseline
    (0, 20.0),
    (2, 130.0),  # stress
    (3, 70.0),  # amusement
    (4, 50.0),  # meditation (excluded from task)
]


def _simulate_chest(duration_s: float, seed: int) -> dict[str, np.ndarray]:
    n = int(duration_s * CHEST_FS)
    ecg = nk.ecg_simulate(duration=int(duration_s), sampling_rate=CHEST_FS,
                          heart_rate=72, random_state=seed)
    eda = nk.eda_simulate(duration=int(duration_s), sampling_rate=CHEST_FS,
                          scr_number=max(2, int(duration_s / 40)), drift=0.01,
                          random_state=seed + 1)
    resp = nk.rsp_simulate(duration=int(duration_s), sampling_rate=CHEST_FS,
                           respiratory_rate=15, random_state=seed + 2)
    return {
        "ECG": np.resize(ecg, n).reshape(-1, 1),
        "EDA": np.resize(eda, n).reshape(-1, 1),
        "Resp": np.resize(resp, n).reshape(-1, 1),
    }


@pytest.fixture(scope="session")
def fake_wesad_zip(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("wesad")
    zip_path = root / "WESAD.zip"
    total_s = sum(d for _, d in WESAD_PROTOCOL)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, sid in enumerate(["S2", "S3"]):
            chest = _simulate_chest(total_s, seed=100 + i)
            n_bvp = int(total_s * 64)
            n_eda_w = int(total_s * 4)
            bvp = nk.ppg_simulate(duration=int(total_s), sampling_rate=64,
                                  heart_rate=72, random_state=200 + i)
            eda_w = nk.eda_simulate(duration=int(total_s), sampling_rate=4,
                                    scr_number=6, random_state=300 + i)
            labels = np.concatenate(
                [np.full(int(d * CHEST_FS), code, dtype=np.int64) for code, d in WESAD_PROTOCOL]
            )
            data = {
                "signal": {
                    "chest": chest,
                    "wrist": {
                        "BVP": np.resize(bvp, n_bvp).reshape(-1, 1),
                        "EDA": np.resize(eda_w, n_eda_w).reshape(-1, 1),
                    },
                },
                "label": labels,
                "subject": sid,
            }
            zf.writestr(f"WESAD/{sid}/{sid}.pkl", pickle.dumps(data, protocol=2))
            zf.writestr(f"WESAD/{sid}/{sid}_readme.txt", "synthetic fixture")
    return zip_path


@pytest.fixture(scope="session")
def fake_maus_rawdir(tmp_path_factory) -> Path:
    """Two subjects: '002' (100 Hz) and '008' (102.5 Hz), six 130 s trials each,
    NaN tail padding of unequal lengths, plus interior NaNs in one trial of '002'."""
    root = tmp_path_factory.mktemp("maus_raw")
    trial_cols = [
        "Trial 1:0back", "Trial 2:2back", "Trial 3:3back",
        "Trial 4:2back", "Trial 5:3back", "Trial 6:0back",
    ]
    for j, (sid, fs) in enumerate([("002", 100.0), ("008", 102.5)]):
        n_per_trial = int(130 * fs)
        pad = 137
        cols = {}
        for t, col in enumerate(trial_cols):
            x = nk.ppg_simulate(duration=131, sampling_rate=int(round(fs)),
                                heart_rate=70 + 4 * t, random_state=17 + 10 * j + t)
            x = np.resize(x, n_per_trial).astype(np.float64)
            full = np.concatenate([x, np.full(pad, np.nan)])
            if sid == "002" and t == 0:
                mid = int(65 * fs)
                full[mid : mid + 5] = np.nan  # interior NaN -> window drops
            cols[col] = full
        (root / sid).mkdir(parents=True)
        pd.DataFrame(cols).to_csv(root / sid / "pixart.csv", index=False)
    return root
