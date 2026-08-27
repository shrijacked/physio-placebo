from __future__ import annotations

import numpy as np

from physio_placebo.data import wesad
from physio_placebo.data.wesad import label_runs_to_segments
from physio_placebo.data.windows import cut_windows


def test_label_runs_to_segments_mapping():
    fs = 10.0
    labels = np.concatenate(
        [
            np.full(50, 0),
            np.full(100, 1),  # baseline  -> y=0
            np.full(30, 0),
            np.full(80, 2),  # stress    -> y=1
            np.full(60, 3),  # amusement -> y=0
            np.full(40, 4),  # meditation-> excluded
        ]
    )
    segs = label_runs_to_segments(labels, fs, positive={2}, negative={1, 3})
    assert [(s.name, s.label) for s in segs] == [
        ("baseline", 0), ("stress", 1), ("amusement", 0),
    ]
    assert segs[0].start_s == 5.0 and segs[0].end_s == 15.0
    assert segs[1].start_s == 18.0 and segs[1].end_s == 26.0


def test_subject_ids_in_fake_zip(fake_wesad_zip):
    assert wesad.subject_ids_in_zip(fake_wesad_zip) == ["S2", "S3"]


def test_end_to_end_windows_from_fake_zip(fake_wesad_zip):
    rec = wesad.load_subject_recording(fake_wesad_zip, "S2")
    assert sorted(rec.channels) == ["bvp", "ecg", "eda", "eda_wrist", "resp"]
    assert rec.fs["ecg"] == 700.0 and rec.fs["bvp"] == 64.0 and rec.fs["eda_wrist"] == 4.0
    # protocol: baseline 130 s, stress 130 s, amusement 70 s -> 3 + 3 + 1 windows
    sw = cut_windows("wesad", "S2", [rec], window_s=60.0, stride_s=30.0)
    assert sw.n_windows == 7
    assert list(sw.y) == [0, 0, 0, 1, 1, 1, 0]
    assert sw.seg_name == ["baseline"] * 3 + ["stress"] * 3 + ["amusement"]
    assert sw.channels["ecg"].shape == (7, 42000)
    assert sw.channels["bvp"].shape == (7, 3840)
    assert sw.channels["eda_wrist"].shape == (7, 240)
    assert sw.n_dropped_nan == 0
