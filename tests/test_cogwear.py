"""CogWear loader tests against the synthetic pilot-cohort fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physio_placebo.data import cogwear
from physio_placebo.data.windows import cut_windows


def test_subject_discovery(fake_cogwear_rawdir):
    assert cogwear.subject_ids_in_dir(fake_cogwear_rawdir) == ["0", "3"]


def test_sessions_labels_and_alignment(fake_cogwear_rawdir):
    recs = cogwear.load_subject_recordings(fake_cogwear_rawdir, "3")
    assert [r.segments[0].name for r in recs] == ["baseline", "cognitive_load"]
    assert [r.segments[0].label for r in recs] == [0, 1]
    for rec in recs:
        assert rec.fs == {"bvp": 64.0, "eda": 4.0}
        # after overlap trimming both channels cover the segment span
        seg = rec.segments[0]
        for name, fs in rec.fs.items():
            assert rec.channels[name].shape[0] >= int(seg.end_s * fs)


def test_windows_end_to_end(fake_cogwear_rawdir):
    recs = cogwear.load_subject_recordings(fake_cogwear_rawdir, "3")
    sw = cut_windows("cogwear", "3", recs, window_s=60.0, stride_s=30.0)
    # baseline ~200 s -> 5 windows; cognitive_load ~310 s -> 9 windows
    counts = dict(zip(*np.unique(sw.y, return_counts=True), strict=True))
    assert counts == {0: 5, 1: 9}
    assert sw.n_dropped_nan == 0
    assert sw.channels["bvp"].shape == (14, 60 * 64)
    assert sw.channels["eda"].shape == (14, 60 * 4)
    assert set(sw.seg_name) == {"baseline", "cognitive_load"}


def test_nan_windows_dropped_and_counted(fake_cogwear_rawdir):
    recs = cogwear.load_subject_recordings(fake_cogwear_rawdir, "0")
    sw = cut_windows("cogwear", "0", recs, window_s=60.0, stride_s=30.0)
    # NaNs at ~150.5 s of cognitive_load EDA kill the windows starting at 120 s and 150 s
    assert sw.n_dropped_nan == 2
    counts = dict(zip(*np.unique(sw.y, return_counts=True), strict=True))
    assert counts == {0: 5, 1: 7}
    for name in ("bvp", "eda"):
        assert np.isfinite(sw.channels[name]).all()


def test_incomplete_subject_excluded(tmp_path):
    # mirrors the real upstream gap: pilot/3 lacks cognitive_load/empatica_eda.csv
    d = tmp_path / "pilot" / "5"
    (d / "baseline").mkdir(parents=True)
    (d / "cognitive_load").mkdir(parents=True)
    for f in ("empatica_bvp.csv", "empatica_eda.csv"):
        (d / "baseline" / f).touch()
    (d / "cognitive_load" / "empatica_bvp.csv").touch()  # EDA missing

    assert cogwear.subject_ids_in_dir(tmp_path) == []


def test_clock_validation_raises(tmp_path):
    d = tmp_path / "pilot" / "0" / "baseline"
    d.mkdir(parents=True)
    n = 400
    bad_t = 1_000_000.0 + np.arange(n) * 0.02  # 50 Hz, not the nominal 64 Hz
    pd.DataFrame({"bvp": np.zeros(n), "time": bad_t}).to_csv(d / "empatica_bvp.csv", index=False)
    good_t = 1_000_000.0 + np.arange(n) * 0.25
    pd.DataFrame({"eda": np.zeros(n), "time": good_t}).to_csv(d / "empatica_eda.csv", index=False)
    (tmp_path / "pilot" / "0" / "cognitive_load").mkdir(parents=True)

    with pytest.raises(ValueError, match="does not match nominal"):
        cogwear.load_subject_recordings(tmp_path, "0")
