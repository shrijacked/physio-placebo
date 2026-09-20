"""MAUS Week-1/2 chain on the fixture distribution (no IEEE DataPort needed).

When real data lands in data/raw/maus/, the same functions run via
scripts/prepare_maus.py → extract_features.py → run_classical_floor.py.
"""

from __future__ import annotations

import pandas as pd

from physio_placebo.baselines.classical import run_classical_floor
from physio_placebo.data import maus
from physio_placebo.data.windows import cut_windows
from physio_placebo.features import frozen

TINY_FLOOR = {"seed": 1337, "n_permutations": 5, "models": ["logreg_l2", "tree_depth3"]}


def test_maus_fixture_windows_features_and_floor(fake_maus_rawdir):
    frames = []
    for sid in maus.subject_ids_in_dir(fake_maus_rawdir):
        recs = maus.load_subject_recordings(fake_maus_rawdir, sid)
        sw = cut_windows("maus", sid, recs, window_s=60.0, stride_s=30.0)
        assert sw.n_windows > 0
        frames.append(frozen.build_feature_frame(sw))
    df = pd.concat(frames, ignore_index=True)
    cols = frozen.feature_columns("maus")
    assert cols == list(frozen.HRV_FEATURES)
    assert set(df["subject"]) == {"002", "008"}
    assert df[cols].notna().any().any()  # at least some finite HRV values

    res = run_classical_floor(df, cols, TINY_FLOOR)
    assert res["n_subjects"] == 2
    assert res["n_windows"] == len(df)
    assert set(res["class_counts"]) == {"0", "1"}
    assert 0.0 <= res["models"]["logreg_l2"]["pooled_macro_f1"] <= 1.0
    assert res["chance"]["permutation"]["logreg_l2"]["n_permutations"] == 5
