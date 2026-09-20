"""Paradigm C: deterministic matplotlib line plot of the primary channel."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from physio_placebo.paradigms.series import primary_channel, window_index
from physio_placebo.paths import configs_dir

C_DATASETS = ("wesad", "maus")


def load_c_spec() -> dict[str, Any]:
    return yaml.safe_load((configs_dir() / "paradigm_c.yaml").read_text())


def assert_c_dataset(dataset: str) -> None:
    if dataset not in C_DATASETS:
        raise ValueError(f"Paradigm C is WESAD and MAUS only, not {dataset!r}")


def render_series_png(x: np.ndarray, *, fs: float, channel: str) -> bytes:
    """Same samples + locked style → same PNG bytes. No subject, no class, no y."""
    spec = load_c_spec()["plot"]
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty series")
    t = np.arange(arr.size, dtype=np.float64) / float(fs)
    fig = Figure(
        figsize=(float(spec["width_in"]), float(spec["height_in"])),
        dpi=int(spec["dpi"]),
        facecolor=str(spec["facecolor"]),
    )
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.12, 0.22, 0.86, 0.70))
    ax.set_facecolor(str(spec["facecolor"]))
    ax.plot(t, arr, color=str(spec["color"]), linewidth=float(spec["linewidth"]), solid_capstyle="butt")
    ax.set_xlabel(str(spec["xlabel"]))
    ax.set_ylabel(str(channel))
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.tick_params(direction="out", length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6, optimize=False)
    return buf.getvalue()


def plot_caption() -> str:
    return str(load_c_spec()["caption"])


def plot_body(
    sw,
    *,
    t0: float,
    seg_name: str,
    dataset: str,
) -> tuple[str, bytes]:
    assert_c_dataset(dataset)
    idx = window_index(sw, t0, seg_name)
    channel = primary_channel(dataset)
    png = render_series_png(
        sw.channels[channel][idx],
        fs=float(sw.fs[channel]),
        channel=channel,
    )
    return plot_caption(), png
