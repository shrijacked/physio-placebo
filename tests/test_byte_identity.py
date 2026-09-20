"""Week-2 exit criterion: classical floor and LLM verbalizer share one matrix.

The LLM scorer (Week 4) must call ``load_locked_feature_matrix`` /
``assert_matches_locked_floor`` so it cannot point at a regenerated parquet.
"""

from __future__ import annotations

import json

import pytest

from physio_placebo.features import frozen
from physio_placebo.features.matrix import (
    assert_matches_locked_floor,
    load_feature_matrix,
    load_locked_feature_matrix,
    locked_floor_registry,
)


@pytest.mark.parametrize("dataset", ["wesad", "cogwear", "maus"])
def test_locked_floor_parquet_is_byte_identical(dataset):
    digest = assert_matches_locked_floor(dataset)
    assert len(digest) == 64


@pytest.mark.parametrize("dataset", ["wesad", "cogwear", "maus"])
def test_canonical_loader_returns_frozen_columns(dataset):
    df = load_locked_feature_matrix(dataset)
    cols = frozen.feature_columns(dataset)
    assert list(df[cols].columns) == cols
    floor = locked_floor_registry()[dataset]
    expected_n = json.loads(floor.read_text())["results"]["n_windows"]
    assert len(df) == expected_n


def test_loader_rejects_missing_feature_column(tmp_path):
    df = load_locked_feature_matrix("cogwear")
    dropped = df.drop(columns=["hrv_sdnn"])
    bad = tmp_path / "bad.parquet"
    dropped.to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="hrv_sdnn"):
        load_feature_matrix(bad, "cogwear")


def test_assert_fails_on_tampered_bytes(tmp_path):
    df = load_locked_feature_matrix("wesad")
    df.loc[0, "hrv_mean_nn"] = 999.0
    tampered = tmp_path / "tampered.parquet"
    df.to_parquet(tampered, index=False)
    with pytest.raises(AssertionError, match="byte-identical"):
        assert_matches_locked_floor("wesad", tampered)
