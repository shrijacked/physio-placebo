"""FROZEN feature extraction.

This module is the single feature definition for the whole project: the classical
baselines (Week 2) and the LLM verbalizer (Paradigm B, Weeks 4+) consume the
matrices produced here — identical inputs by construction.

Freeze discipline: at the end of Week 1 the SHA-256 of this file's source bytes
plus the pinned neurokit2/numpy/scipy versions are recorded in
``configs/frozen_features.lock.json`` (see ``scripts/freeze_features.py``).
``tests/test_feature_hash.py`` fails if this file changes afterwards. Do not edit
without explicit approval; if approved, re-freeze and regenerate all matrices.

Feature families
----------------
- HRV time domain (from ECG R-peaks or PPG systolic peaks): mean NN, SDNN, RMSSD,
  pNN50, mean HR.
- HRV frequency domain: LF (0.04-0.15 Hz), HF (0.15-0.40 Hz) power and LF/HF from a
  Welch PSD of the 4 Hz-interpolated tachogram. Caveat (documented, standard in this
  literature): on 60 s windows the LF estimate is at the edge of resolvability.
- EDA (cvxEDA decomposition at 8 Hz): tonic mean/std, phasic std, SCR count and mean
  amplitude (scipy peak picking on the phasic component), raw level mean/std.
- Respiration: breath rate mean/std and mean peak-to-trough amplitude.

NaN policy: if a modality fails on a window (no detectable beats, solver failure),
that window's features for the modality are NaN. NaNs are never imputed here;
imputation happens inside model pipelines with train-fold-only statistics.
NaN rates are reported by the extraction script.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal

HRV_FEATURES = [
    "hrv_mean_nn", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_mean_hr",
    "hrv_lf", "hrv_hf", "hrv_lf_hf",
]
EDA_FEATURES = [
    "eda_tonic_mean", "eda_tonic_std", "eda_phasic_std",
    "eda_scr_count", "eda_scr_amp_mean", "eda_level_mean", "eda_level_std",
]
RESP_FEATURES = ["resp_rate_mean", "resp_rate_std", "resp_amp_mean"]

# Which channel feeds which feature family, per dataset. WESAD HRV comes from chest
# ECG (its gold-standard cardiac channel); CogWear from E4 BVP; MAUS from PixArt
# wrist PPG (the spec's canonical MAUS signal). One EDA source per dataset: WESAD
# chest EDA, CogWear wrist EDA. Respiration exists only in WESAD.
DATASET_FEATURE_PLAN: dict[str, dict[str, str]] = {
    "wesad": {"hrv_ecg": "ecg", "eda": "eda", "resp": "resp"},
    "cogwear": {"hrv_ppg": "bvp", "eda": "eda"},
    "maus": {"hrv_ppg": "ppg"},
}

_EDA_TARGET_FS = 8.0
_TACHOGRAM_FS = 4.0
_LF_BAND = (0.04, 0.15)
_HF_BAND = (0.15, 0.40)


def feature_columns(dataset: str) -> list[str]:
    plan = DATASET_FEATURE_PLAN[dataset]
    cols: list[str] = []
    if "hrv_ecg" in plan or "hrv_ppg" in plan:
        cols += HRV_FEATURES
    if "eda" in plan:
        cols += EDA_FEATURES
    if "resp" in plan:
        cols += RESP_FEATURES
    return cols


def _nan_features(names: list[str]) -> dict[str, float]:
    return {n: float("nan") for n in names}


def _hrv_from_peak_indices(peaks: np.ndarray, fs: float) -> dict[str, float]:
    out = _nan_features(HRV_FEATURES)
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.size < 5:
        return out
    ibi_ms = np.diff(peaks) / fs * 1000.0
    # Physiological plausibility gate: drop windows whose beat detection is clearly
    # broken (IBI outside 250-2000 ms), rather than emitting garbage numbers.
    if np.any(ibi_ms < 250.0) or np.any(ibi_ms > 2000.0):
        ibi_ms = ibi_ms[(ibi_ms >= 250.0) & (ibi_ms <= 2000.0)]
        if ibi_ms.size < 4:
            return out
    out["hrv_mean_nn"] = float(np.mean(ibi_ms))
    out["hrv_sdnn"] = float(np.std(ibi_ms, ddof=1))
    diffs = np.diff(ibi_ms)
    if diffs.size:
        out["hrv_rmssd"] = float(np.sqrt(np.mean(diffs**2)))
        out["hrv_pnn50"] = float(np.mean(np.abs(diffs) > 50.0) * 100.0)
    out["hrv_mean_hr"] = float(60000.0 / np.mean(ibi_ms))

    # Frequency domain on the uniformly resampled tachogram.
    beat_times_s = np.cumsum(ibi_ms) / 1000.0
    span = beat_times_s[-1] - beat_times_s[0]
    if span >= 20.0 and ibi_ms.size >= 8:
        grid = np.arange(beat_times_s[0], beat_times_s[-1], 1.0 / _TACHOGRAM_FS)
        tach = np.interp(grid, beat_times_s, ibi_ms)
        tach = tach - np.mean(tach)
        nperseg = min(tach.size, 256)
        freqs, psd = scipy.signal.welch(tach, fs=_TACHOGRAM_FS, nperseg=nperseg)
        lf_mask = (freqs >= _LF_BAND[0]) & (freqs < _LF_BAND[1])
        hf_mask = (freqs >= _HF_BAND[0]) & (freqs < _HF_BAND[1])
        if lf_mask.any() and hf_mask.any():
            lf = float(np.trapezoid(psd[lf_mask], freqs[lf_mask]))
            hf = float(np.trapezoid(psd[hf_mask], freqs[hf_mask]))
            out["hrv_lf"] = lf
            out["hrv_hf"] = hf
            out["hrv_lf_hf"] = lf / hf if hf > 0 else float("nan")
    return out


def hrv_features_from_ecg(x: np.ndarray, fs: float) -> dict[str, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(np.asarray(x, dtype=np.float64), sampling_rate=int(fs))
            _, info = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
        return _hrv_from_peak_indices(np.asarray(info["ECG_R_Peaks"]), fs)
    except Exception:
        return _nan_features(HRV_FEATURES)


def hrv_features_from_ppg(x: np.ndarray, fs: float) -> dict[str, float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ppg_clean(np.asarray(x, dtype=np.float64), sampling_rate=int(fs))
            info = nk.ppg_findpeaks(cleaned, sampling_rate=int(fs))
        return _hrv_from_peak_indices(np.asarray(info["PPG_Peaks"]), fs)
    except Exception:
        return _nan_features(HRV_FEATURES)


def eda_features(x: np.ndarray, fs: float) -> dict[str, float]:
    out = _nan_features(EDA_FEATURES)
    try:
        x = np.asarray(x, dtype=np.float64)
        out["eda_level_mean"] = float(np.mean(x))
        out["eda_level_std"] = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        if fs > _EDA_TARGET_FS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = nk.signal_resample(
                    x, sampling_rate=int(fs), desired_sampling_rate=int(_EDA_TARGET_FS),
                    method="interpolation",
                )
            fs_eff = _EDA_TARGET_FS
        else:
            fs_eff = fs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomp = nk.eda_phasic(x, sampling_rate=int(fs_eff), method="cvxeda")
        tonic = decomp["EDA_Tonic"].to_numpy()
        phasic = decomp["EDA_Phasic"].to_numpy()
        out["eda_tonic_mean"] = float(np.mean(tonic))
        out["eda_tonic_std"] = float(np.std(tonic, ddof=1))
        out["eda_phasic_std"] = float(np.std(phasic, ddof=1))
        # SCR events: peaks on the phasic drive with a minimum 0.01 uS prominence
        # (conventional threshold) and >= 1 s separation.
        peaks, props = scipy.signal.find_peaks(
            phasic, prominence=0.01, distance=max(1, int(fs_eff * 1.0))
        )
        out["eda_scr_count"] = float(peaks.size)
        out["eda_scr_amp_mean"] = float(np.mean(props["prominences"])) if peaks.size else 0.0
        return out
    except Exception:
        return out if np.isfinite(out["eda_level_mean"]) else _nan_features(EDA_FEATURES)


def resp_features(x: np.ndarray, fs: float) -> dict[str, float]:
    out = _nan_features(RESP_FEATURES)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.rsp_clean(np.asarray(x, dtype=np.float64), sampling_rate=int(fs))
            _, info = nk.rsp_peaks(cleaned, sampling_rate=int(fs))
        peaks = np.asarray(info["RSP_Peaks"], dtype=np.int64)
        troughs = np.asarray(info["RSP_Troughs"], dtype=np.int64)
        if peaks.size < 3:
            return out
        intervals_s = np.diff(peaks) / fs
        intervals_s = intervals_s[(intervals_s > 1.0) & (intervals_s < 20.0)]
        if intervals_s.size < 2:
            return out
        rates = 60.0 / intervals_s
        out["resp_rate_mean"] = float(np.mean(rates))
        out["resp_rate_std"] = float(np.std(rates, ddof=1))
        n = min(peaks.size, troughs.size)
        if n:
            out["resp_amp_mean"] = float(np.mean(cleaned[peaks[:n]] - cleaned[troughs[:n]]))
        return out
    except Exception:
        return out


def extract_window_features(
    channels: dict[str, np.ndarray],
    fs: dict[str, float],
    dataset: str,
) -> dict[str, float]:
    plan = DATASET_FEATURE_PLAN[dataset]
    out: dict[str, float] = {}
    if "hrv_ecg" in plan:
        ch = plan["hrv_ecg"]
        out.update(hrv_features_from_ecg(channels[ch], fs[ch]))
    if "hrv_ppg" in plan:
        ch = plan["hrv_ppg"]
        out.update(hrv_features_from_ppg(channels[ch], fs[ch]))
    if "eda" in plan:
        ch = plan["eda"]
        out.update(eda_features(channels[ch], fs[ch]))
    if "resp" in plan:
        ch = plan["resp"]
        out.update(resp_features(channels[ch], fs[ch]))
    return out


def build_feature_frame(sw) -> pd.DataFrame:
    """Feature rows for one subject's windows (see data.windows.SubjectWindows)."""
    rows = []
    for i in range(sw.n_windows):
        window = {ch: sw.channels[ch][i] for ch in sw.channels}
        feats = extract_window_features(window, sw.fs, sw.dataset)
        rows.append(
            {
                "dataset": sw.dataset,
                "subject": sw.subject,
                "window_id": f"{sw.subject}:{sw.seg_name[i]}:{sw.t0[i]:.1f}",
                "seg_name": sw.seg_name[i],
                "t0": float(sw.t0[i]),
                "y": int(sw.y[i]),
                **feats,
            }
        )
    return pd.DataFrame(rows)


def frozen_code_hash() -> dict[str, str]:
    """Identity of the frozen feature code: source hash + numeric-library versions."""
    import neurokit2
    import scipy

    return {
        "frozen_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "neurokit2": neurokit2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
