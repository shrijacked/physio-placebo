"""Paradigm C plots: same window → same PNG bytes. No GPU."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from physio_placebo.data.windows import SubjectWindows
from physio_placebo.paradigms.plot import (
    C_DATASETS,
    assert_c_dataset,
    plot_body,
    plot_caption,
    render_series_png,
)


def _png_text_blob(png: bytes) -> bytes:
    info = Image.open(io.BytesIO(png)).info
    return " ".join(f"{k}={v}" for k, v in info.items()).encode()


def test_same_series_same_png_bytes():
    x = np.sin(np.linspace(0, 8 * np.pi, 400, dtype=np.float64))
    a = render_series_png(x, fs=64.0, channel="ecg")
    b = render_series_png(x, fs=64.0, channel="ecg")
    assert a == b
    assert a.startswith(b"\x89PNG")
    assert len(a) > 200


def test_different_series_different_png_bytes():
    t = np.linspace(0, 8 * np.pi, 400, dtype=np.float64)
    a = render_series_png(np.sin(t), fs=64.0, channel="ecg")
    b = render_series_png(np.cos(t), fs=64.0, channel="ecg")
    assert a != b


def test_png_does_not_embed_label_or_subject():
    x = np.linspace(-1, 1, 128, dtype=np.float64)
    png = render_series_png(x, fs=4.0, channel="ppg")
    meta = _png_text_blob(png)
    assert b"stress" not in meta
    assert b"S2" not in meta
    assert b"non-stress" not in meta
    assert Image.open(io.BytesIO(png)).info.get("Software", "") in {"", None}


def test_cogwear_rejected():
    with pytest.raises(ValueError, match="MAUS"):
        assert_c_dataset("cogwear")
    assert C_DATASETS == ("wesad", "maus")


def test_plot_body_uses_caption_and_primary_channel():
    n = 256
    sw = SubjectWindows(
        dataset="wesad",
        subject="S99",
        channels={"ecg": np.linspace(0, 1, n, dtype=np.float32).reshape(1, n)},
        fs={"ecg": 4.0},
        y=np.array([1], dtype=np.int8),
        t0=np.array([0.0]),
        seg_name=["stress"],
    )
    caption, png = plot_body(sw, t0=0.0, seg_name="stress", dataset="wesad")
    assert caption == plot_caption()
    assert png.startswith(b"\x89PNG")
    meta = _png_text_blob(png)
    assert b"S99" not in meta
    assert b"stress" not in meta
