from __future__ import annotations

import numpy as np
import pytest

from physio_placebo.data.windows import (
    Recording,
    Segment,
    cut_windows,
    load_subject_windows,
    save_subject_windows,
)


def _recording(n_seconds: float = 200.0) -> Recording:
    fs = {"a": 10.0, "b": 4.0}
    channels = {
        "a": np.arange(int(n_seconds * fs["a"]), dtype=np.float64),
        "b": np.arange(int(n_seconds * fs["b"]), dtype=np.float64),
    }
    segments = [Segment(0.0, 130.0, 0, "rest"), Segment(130.0, 200.0, 1, "task")]
    return Recording(channels=channels, fs=fs, segments=segments)


def test_window_counts_and_labels():
    sw = cut_windows("wesad", "SX", [_recording()], window_s=60.0, stride_s=30.0)
    # rest 130 s -> t0 in {0, 30, 60}; task 70 s -> t0 in {130}
    assert sw.n_windows == 4
    assert list(sw.t0) == [0.0, 30.0, 60.0, 130.0]
    assert list(sw.y) == [0, 0, 0, 1]
    assert sw.seg_name == ["rest", "rest", "rest", "task"]
    assert sw.channels["a"].shape == (4, 600)
    assert sw.channels["b"].shape == (4, 240)
    # window content: purity via value ranges (channel 'a' is a ramp of sample indices)
    assert sw.channels["a"][3, 0] == pytest.approx(1300.0)


def test_nan_windows_dropped_and_counted():
    rec = _recording()
    rec.channels["a"][int(35 * 10)] = np.nan  # inside t0=0 and t0=30 windows only
    sw = cut_windows("wesad", "SX", [rec], window_s=60.0, stride_s=30.0)
    assert sw.n_windows == 2
    assert sw.n_dropped_nan == 2
    assert list(sw.t0) == [60.0, 130.0]


def test_segment_shorter_than_window_produces_nothing():
    rec = _recording()
    rec.segments = [Segment(0.0, 45.0, 1, "short")]
    sw = cut_windows("wesad", "SX", [rec], window_s=60.0, stride_s=30.0)
    assert sw.n_windows == 0


def test_multiple_recordings_accumulate():
    sw = cut_windows("maus", "002", [_recording(), _recording()], 60.0, 30.0)
    assert sw.n_windows == 8


def test_inconsistent_channels_rejected():
    r1 = _recording()
    r2 = _recording()
    r2.fs["a"] = 11.0
    with pytest.raises(ValueError, match="inconsistent"):
        cut_windows("maus", "002", [r1, r2], 60.0, 30.0)


def test_save_load_roundtrip(tmp_path):
    sw = cut_windows("wesad", "SX", [_recording()], 60.0, 30.0)
    npz = save_subject_windows(sw, tmp_path, extra_meta={"origin": "test"})
    back = load_subject_windows(npz)
    assert back.subject == sw.subject and back.dataset == sw.dataset
    assert back.fs == sw.fs
    np.testing.assert_array_equal(back.y, sw.y)
    np.testing.assert_array_equal(back.t0, sw.t0)
    assert back.seg_name == sw.seg_name
    for ch in sw.channels:
        np.testing.assert_array_equal(back.channels[ch], sw.channels[ch])
