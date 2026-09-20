"""Paradigm A: downsampled primary-channel series as prompt body text."""

from __future__ import annotations

import numpy as np
import yaml

from physio_placebo.paradigms.downsample import apply_scheme
from physio_placebo.paths import configs_dir, data_processed
from physio_placebo.data.windows import SubjectWindows, load_subject_windows


def load_sweep_spec() -> dict:
    return yaml.safe_load((configs_dir() / "prompt_sweep.yaml").read_text())


def primary_channel(dataset: str) -> str:
    return str(load_sweep_spec()["primary_channel"][dataset])


def load_windows(dataset: str, subject: str) -> SubjectWindows:
    return load_subject_windows(data_processed(dataset) / f"{subject}.npz")


def window_index(sw: SubjectWindows, t0: float, seg_name: str) -> int:
    for i in range(sw.n_windows):
        if sw.seg_name[i] == seg_name and abs(float(sw.t0[i]) - float(t0)) < 1e-3:
            return i
    raise KeyError(f"no window {sw.subject}:{seg_name}:{t0} in {sw.dataset}")


def format_series(
    x: np.ndarray,
    *,
    channel: str,
    fs: float,
    scheme: str,
) -> str:
    xs = apply_scheme(np.asarray(x, dtype=float), scheme)
    values = " ".join(f"{v:.4g}" for v in xs)
    return f"{channel} fs={fs:g} Hz n={xs.size} scheme={scheme}: {values}"


def series_body(
    sw: SubjectWindows,
    *,
    t0: float,
    seg_name: str,
    scheme: str,
    dataset: str,
) -> str:
    idx = window_index(sw, t0, seg_name)
    channel = primary_channel(dataset)
    return format_series(
        sw.channels[channel][idx],
        channel=channel,
        fs=float(sw.fs[channel]),
        scheme=scheme,
    )
