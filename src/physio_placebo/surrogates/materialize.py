"""Build surrogate windows and feature matrices from the locked real windows.

Subject order is the sorted subject id. A permutation moves every channel of a
window together. The target window keeps its label and its id; only the signal
changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from physio_placebo.data.windows import SubjectWindows, load_subject_windows, save_subject_windows
from physio_placebo.features.frozen import build_feature_frame, feature_columns
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.paradigms.series import load_windows
from physio_placebo.paths import data_processed
from physio_placebo.surrogates.ladder import (
    ALL_SURROGATES,
    iaaft_batch,
    permutation_donors,
)

SURROGATE_SEED = 1337


def subject_ids(dataset: str) -> list[str]:
    return sorted(p.stem for p in data_processed(dataset).glob("*.npz"))


def load_ordered(dataset: str) -> list[SubjectWindows]:
    ids = subject_ids(dataset)
    if not ids:
        raise FileNotFoundError(f"no windows for {dataset} under {data_processed(dataset)}")
    return [load_windows(dataset, subject) for subject in ids]


def _labels(sws: Sequence[SubjectWindows]) -> tuple[np.ndarray, list[str]]:
    y = np.concatenate([np.asarray(sw.y) for sw in sws])
    subjects: list[str] = []
    for sw in sws:
        subjects.extend([sw.subject] * sw.n_windows)
    return y, subjects


def window_id(subject: str, seg_name: str, t0: float) -> str:
    return f"{subject}:{seg_name}:{t0:.1f}"


def _real_feature_frame(dataset: str) -> pd.DataFrame:
    frame = load_locked_feature_matrix(dataset).copy()
    if "window_id" not in frame.columns:
        frame["window_id"] = [
            window_id(str(s), str(seg), float(t0))
            for s, seg, t0 in zip(frame["subject"], frame["seg_name"], frame["t0"], strict=True)
        ]
    return frame


def permuted_features(dataset: str, *, preserve_label: bool, seed: int) -> pd.DataFrame:
    """Donor features paired with the target window's label and id."""
    sws = load_ordered(dataset)
    y, subjects = _labels(sws)
    donors = permutation_donors(subjects, y, preserve_label=preserve_label, seed=seed)
    real = _real_feature_frame(dataset)
    order = [
        window_id(sw.subject, sw.seg_name[i], float(sw.t0[i]))
        for sw in sws
        for i in range(sw.n_windows)
    ]
    real = real.set_index("window_id", drop=False)
    missing = [wid for wid in order if wid not in real.index]
    if missing:
        raise ValueError(f"{dataset} feature matrix is missing {len(missing)} window ids")
    ordered = real.loc[order]
    donor_rows = ordered.iloc[donors]
    cols = feature_columns(dataset)
    out = ordered.copy()
    for col in cols:
        out[col] = donor_rows[col].to_numpy()
    return out.reset_index(drop=True)


def _sample_moments(sws: Sequence[SubjectWindows]) -> dict[str, tuple[float, float]]:
    moments = {}
    names = sws[0].channels
    for name in names:
        total = 0.0
        total_sq = 0.0
        count = 0
        for sw in sws:
            values = np.asarray(sw.channels[name], dtype=float)
            total += float(values.sum())
            total_sq += float(np.square(values).sum())
            count += int(values.size)
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        moments[name] = (mean, variance)
    return moments


def _resample_row(row: np.ndarray, n_samples: int) -> np.ndarray:
    if row.shape[0] == n_samples:
        return np.asarray(row, dtype=float)
    src = np.linspace(0.0, 1.0, row.shape[0])
    dst = np.linspace(0.0, 1.0, n_samples)
    return np.interp(dst, src, np.asarray(row, dtype=float))


def _locate(spans: list[tuple[int, int]], flat: int) -> tuple[int, int]:
    for subject_i, (start, stop) in enumerate(spans):
        if start <= flat < stop:
            return subject_i, flat - start
    raise IndexError(flat)


def write_signal_surrogates(dataset: str, kind: str, out_dir, *, seed: int = SURROGATE_SEED) -> list[str]:
    """Write per-subject npz files. ``deleted`` writes nothing and returns [].

    Windows inside one subject share a length. MAUS does not: some subjects
    were recorded at 100 Hz and some at 102.5 Hz. A donor row from the other
    rate is linearly resampled onto the target length so the window stays one
    minute. Features for permutations are still the donor's original feature
    row, not features of the resampled waveform.
    """
    if kind not in ALL_SURROGATES:
        raise ValueError(kind)
    if kind == "deleted":
        return []
    sws = load_ordered(dataset)
    names = list(sws[0].channels)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sw in sws:
        spans.append((cursor, cursor + sw.n_windows))
        cursor += sw.n_windows
    rng = np.random.default_rng(seed)
    donors = None
    moments = None
    if kind in ("subject_perm_broken", "subject_perm_preserved"):
        y, subjects = _labels(sws)
        donors = permutation_donors(
            subjects,
            y,
            preserve_label=kind == "subject_perm_preserved",
            seed=seed,
        )
    elif kind in ("gaussian", "flatline"):
        moments = _sample_moments(sws)

    written = []
    for subject_i, sw in enumerate(sws):
        built: dict[str, np.ndarray] = {}
        if donors is not None:
            start, _stop = spans[subject_i]
            for name in names:
                rows = []
                width = sw.channels[name].shape[1]
                for offset, donor_flat in enumerate(donors[start : start + sw.n_windows]):
                    donor_subject, donor_local = _locate(spans, int(donor_flat))
                    donor_row = sws[donor_subject].channels[name][donor_local]
                    rows.append(_resample_row(donor_row, width))
                    del offset
                built[name] = np.stack(rows).astype(np.float32)
        elif kind == "iaaft":
            for name in names:
                built[name] = iaaft_batch(np.asarray(sw.channels[name], dtype=float), rng).astype(
                    np.float32
                )
        elif kind == "gaussian":
            assert moments is not None
            for name in names:
                mean, variance = moments[name]
                built[name] = rng.normal(mean, np.sqrt(variance), size=sw.channels[name].shape).astype(
                    np.float32
                )
        elif kind == "flatline":
            assert moments is not None
            for name in names:
                mean, _variance = moments[name]
                built[name] = np.full(sw.channels[name].shape, mean, dtype=np.float32)
        else:
            raise ValueError(kind)
        save_subject_windows(
            SubjectWindows(
                dataset=sw.dataset,
                subject=sw.subject,
                channels=built,
                fs=dict(sw.fs),
                y=np.asarray(sw.y),
                t0=np.asarray(sw.t0),
                seg_name=list(sw.seg_name),
                n_dropped_nan=sw.n_dropped_nan,
            ),
            out_dir,
            extra_meta={"surrogate": kind, "seed": seed},
        )
        written.append(sw.subject)
    return written


def features_from_windows(dataset: str, windows_dir) -> pd.DataFrame:
    frames = []
    for path in sorted(windows_dir.glob("*.npz")):
        sw = load_subject_windows(path)
        frames.append(build_feature_frame(sw))
    if not frames:
        raise FileNotFoundError(f"no windows in {windows_dir}")
    return pd.concat(frames, ignore_index=True)
