"""Shared windowing data model.

A *recording* is a set of synchronised channels on one local clock plus labeled
segments on that clock. A loader returns one or more recordings per subject
(WESAD: one continuous protocol recording; MAUS: one recording per n-back trial;
CogWear: one recording per session).

Windows are cut strictly inside a single labeled segment, so every window has
purity 1.0 by construction. Windows containing NaNs in any channel are dropped
and counted (`n_dropped_nan`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    label: int  # binary task label, 0 or 1
    name: str  # human-readable protocol name, e.g. "stress", "trial_2_2back"


@dataclass
class Recording:
    channels: dict[str, np.ndarray]  # channel name -> 1-D signal
    fs: dict[str, float]  # channel name -> sampling rate (Hz)
    segments: list[Segment]


@dataclass
class SubjectWindows:
    dataset: str
    subject: str
    channels: dict[str, np.ndarray]  # channel name -> (n_windows, n_samples_channel) float32
    fs: dict[str, float]
    y: np.ndarray  # (n_windows,) int8
    t0: np.ndarray  # (n_windows,) float64, window start on the recording-local clock
    seg_name: list[str] = field(default_factory=list)
    n_dropped_nan: int = 0

    @property
    def n_windows(self) -> int:
        return int(self.y.shape[0])


def _window_starts(seg: Segment, window_s: float, stride_s: float) -> list[float]:
    starts = []
    t = seg.start_s
    while t + window_s <= seg.end_s + 1e-9:
        starts.append(t)
        t += stride_s
    return starts


def cut_windows(
    dataset: str,
    subject: str,
    recordings: list[Recording],
    window_s: float,
    stride_s: float,
) -> SubjectWindows:
    """Cut fixed-length windows from labeled segments across all recordings."""
    if not recordings:
        raise ValueError("no recordings supplied")
    channel_names = sorted(recordings[0].channels)
    fs = dict(recordings[0].fs)
    for rec in recordings:
        if sorted(rec.channels) != channel_names or {k: rec.fs[k] for k in rec.fs} != fs:
            raise ValueError(f"inconsistent channels/fs across recordings for {subject}")

    n_samples = {ch: int(round(window_s * fs[ch])) for ch in channel_names}
    per_channel: dict[str, list[np.ndarray]] = {ch: [] for ch in channel_names}
    ys: list[int] = []
    t0s: list[float] = []
    seg_names: list[str] = []
    n_dropped = 0

    for rec in recordings:
        for seg in rec.segments:
            for t0 in _window_starts(seg, window_s, stride_s):
                slices = {}
                ok = True
                for ch in channel_names:
                    i0 = int(round(t0 * fs[ch]))
                    i1 = i0 + n_samples[ch]
                    x = rec.channels[ch]
                    if i1 > x.shape[0]:
                        ok = False
                        break
                    w = x[i0:i1]
                    if not np.all(np.isfinite(w)):
                        ok = False
                        break
                    slices[ch] = w.astype(np.float32)
                if not ok:
                    n_dropped += 1
                    continue
                for ch in channel_names:
                    per_channel[ch].append(slices[ch])
                ys.append(seg.label)
                t0s.append(t0)
                seg_names.append(seg.name)

    channels = {
        ch: (np.stack(v) if v else np.empty((0, n_samples[ch]), dtype=np.float32))
        for ch, v in per_channel.items()
    }
    return SubjectWindows(
        dataset=dataset,
        subject=subject,
        channels=channels,
        fs=fs,
        y=np.asarray(ys, dtype=np.int8),
        t0=np.asarray(t0s, dtype=np.float64),
        seg_name=seg_names,
        n_dropped_nan=n_dropped,
    )


def save_subject_windows(sw: SubjectWindows, out_dir: Path, extra_meta: dict | None = None) -> Path:
    """Persist windows as <subject>.npz plus a <subject>.meta.json sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sw.subject}.npz"
    arrays = {f"ch::{name}": arr for name, arr in sw.channels.items()}
    arrays["y"] = sw.y
    arrays["t0"] = sw.t0
    arrays["seg_name"] = np.asarray(sw.seg_name)
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "dataset": sw.dataset,
        "subject": sw.subject,
        "fs": sw.fs,
        "n_windows": sw.n_windows,
        "n_dropped_nan": sw.n_dropped_nan,
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(sw.y, return_counts=True), strict=True)}
        if sw.n_windows
        else {},
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / f"{sw.subject}.meta.json").write_text(json.dumps(meta, indent=2))
    return npz_path


def load_subject_windows(npz_path: Path) -> SubjectWindows:
    meta = json.loads((npz_path.parent / f"{npz_path.stem}.meta.json").read_text())
    with np.load(npz_path, allow_pickle=False) as z:
        channels = {k.removeprefix("ch::"): z[k] for k in z.files if k.startswith("ch::")}
        y = z["y"]
        t0 = z["t0"]
        seg_name = [str(s) for s in z["seg_name"]]
    return SubjectWindows(
        dataset=meta["dataset"],
        subject=meta["subject"],
        channels=channels,
        fs={k: float(v) for k, v in meta["fs"].items()},
        y=y,
        t0=t0,
        seg_name=seg_name,
        n_dropped_nan=int(meta.get("n_dropped_nan", 0)),
    )
