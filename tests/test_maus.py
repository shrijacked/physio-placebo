from __future__ import annotations

from physio_placebo.data import maus
from physio_placebo.data.windows import cut_windows


def test_subject_discovery(fake_maus_rawdir):
    assert maus.subject_ids_in_dir(fake_maus_rawdir) == ["002", "008"]


def test_pixart_fs_rule():
    for sid in ["002", "003", "004", "005", "006"]:
        assert maus.pixart_fs(sid) == 100.0
    for sid in ["008", "010", "025"]:
        assert maus.pixart_fs(sid) == 102.5


def test_trial_labels_and_windows(fake_maus_rawdir):
    recs = maus.load_subject_recordings(fake_maus_rawdir, "008")
    assert len(recs) == 6
    names = [r.segments[0].name for r in recs]
    labels = [r.segments[0].label for r in recs]
    assert names == [
        "trial_1_0back", "trial_2_2back", "trial_3_3back",
        "trial_4_2back", "trial_5_3back", "trial_6_0back",
    ]
    assert labels == [0, 1, 1, 1, 1, 0]
    sw = cut_windows("maus", "008", recs, window_s=60.0, stride_s=30.0)
    # 130 s per trial -> 3 windows each, 6 trials -> 18 windows; 0-back trials give 6 y=0
    assert sw.n_windows == 18
    assert int((sw.y == 0).sum()) == 6 and int((sw.y == 1).sum()) == 12
    assert sw.channels["ppg"].shape == (18, int(round(60.0 * 102.5)))


def test_interior_nan_drops_windows(fake_maus_rawdir):
    recs = maus.load_subject_recordings(fake_maus_rawdir, "002")
    sw = cut_windows("maus", "002", recs, window_s=60.0, stride_s=30.0)
    # subject 002 trial 1 has interior NaNs around t=65 s -> windows at t0=30 (covers
    # 30-90) and t0=60 (covers 60-120) are dropped; t0=0 survives
    assert sw.n_dropped_nan == 2
    assert sw.n_windows == 16
