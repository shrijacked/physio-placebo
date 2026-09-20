"""Paradigm A series compression. Two documented schemes, no silent default winner."""

from __future__ import annotations

import numpy as np
import yaml

from physio_placebo.paths import configs_dir


def load_downsample_spec() -> dict:
    return yaml.safe_load((configs_dir() / "paradigm_a_downsample.yaml").read_text())


def uniform_stride(x: np.ndarray, n_keep: int) -> np.ndarray:
    """Keep ``n_keep`` samples at even stride; keep all if the series is shorter."""
    arr = np.asarray(x).ravel()
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    if arr.size <= n_keep:
        return arr.copy()
    idx = np.linspace(0, arr.size - 1, num=n_keep)
    return arr[np.round(idx).astype(int)]


def bin_mean(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Non-overlapping block means. Tail samples past a full bin are dropped."""
    arr = np.asarray(x, dtype=float).ravel()
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if arr.size < n_bins:
        return arr.copy()
    width = arr.size // n_bins
    clipped = arr[: width * n_bins]
    return clipped.reshape(n_bins, width).mean(axis=1)


def apply_scheme(x: np.ndarray, scheme: str, spec: dict | None = None) -> np.ndarray:
    cfg = spec if spec is not None else load_downsample_spec()
    if scheme == "uniform_stride":
        return uniform_stride(x, int(cfg["schemes"]["uniform_stride"]["n_keep"]))
    if scheme == "bin_mean":
        return bin_mean(x, int(cfg["schemes"]["bin_mean"]["n_bins"]))
    raise ValueError(f"unknown downsample scheme {scheme!r}")
