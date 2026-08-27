from __future__ import annotations

import neurokit2 as nk
import numpy as np
import pytest

from physio_placebo.features import frozen


@pytest.fixture(scope="module")
def wesad_window():
    fs_ecg = 700
    return (
        {
            "ecg": nk.ecg_simulate(duration=60, sampling_rate=fs_ecg, heart_rate=72, random_state=7),
            "eda": nk.eda_simulate(duration=60, sampling_rate=fs_ecg, scr_number=3, random_state=8),
            "resp": nk.rsp_simulate(duration=60, sampling_rate=fs_ecg, respiratory_rate=15,
                                    random_state=9),
            "bvp": nk.ppg_simulate(duration=60, sampling_rate=64, heart_rate=72, random_state=10),
            "eda_wrist": nk.eda_simulate(duration=60, sampling_rate=4, scr_number=3, random_state=11),
        },
        {"ecg": 700.0, "eda": 700.0, "resp": 700.0, "bvp": 64.0, "eda_wrist": 4.0},
    )


def test_wesad_feature_set_and_plausibility(wesad_window):
    channels, fs = wesad_window
    feats = frozen.extract_window_features(channels, fs, "wesad")
    assert sorted(feats) == sorted(frozen.feature_columns("wesad"))
    assert 40 <= feats["hrv_mean_hr"] <= 120
    assert feats["hrv_sdnn"] >= 0
    assert 6 <= feats["resp_rate_mean"] <= 30
    for k in frozen.EDA_FEATURES:
        assert np.isfinite(feats[k]), k


def test_extraction_is_deterministic(wesad_window):
    channels, fs = wesad_window
    a = frozen.extract_window_features(channels, fs, "wesad")
    b = frozen.extract_window_features(channels, fs, "wesad")
    assert a == b


def test_maus_plan_is_ppg_only():
    x = nk.ppg_simulate(duration=60, sampling_rate=100, heart_rate=68, random_state=3)
    feats = frozen.extract_window_features({"ppg": x}, {"ppg": 100.0}, "maus")
    assert sorted(feats) == sorted(frozen.HRV_FEATURES)
    assert 40 <= feats["hrv_mean_hr"] <= 120


def test_degenerate_signal_yields_nans_not_garbage():
    flat = np.zeros(60 * 700)
    feats = frozen.hrv_features_from_ecg(flat, 700.0)
    assert all(not np.isfinite(v) for v in feats.values())


def test_frozen_code_hash_shape():
    ident = frozen.frozen_code_hash()
    assert set(ident) == {"frozen_py_sha256", "neurokit2", "numpy", "scipy"}
    assert len(ident["frozen_py_sha256"]) == 64
