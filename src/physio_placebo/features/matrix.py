"""Canonical feature-matrix load path.

The Week-2 classical floor and the LLM verbalizer (Paradigm B) both load features
through ``load_feature_matrix``. ``tests/test_byte_identity.py`` asserts that the
locked floor.json SHA-256 still matches the parquet on disk, so a later LLM run
cannot silently point at a different matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from physio_placebo.features import frozen
from physio_placebo.paths import REPO_ROOT, configs_dir
from physio_placebo.provenance import sha256_file


def locked_floor_registry() -> dict[str, Path]:
    raw = yaml.safe_load((configs_dir() / "locked_floors.yaml").read_text())
    return {dataset: REPO_ROOT / rel for dataset, rel in raw.items()}


def load_floor_json(dataset: str) -> dict:
    path = locked_floor_registry()[dataset]
    return json.loads(path.read_text())


def load_feature_matrix(parquet_path: Path | str, dataset: str) -> pd.DataFrame:
    """Load a frozen feature parquet and check it has the expected columns.

    Column order of the feature block is the frozen plan order. Extra columns
    (ids, t0, seg_name) are kept; missing feature columns raise.
    """
    path = Path(parquet_path)
    df = pd.read_parquet(path)
    feat_cols = frozen.feature_columns(dataset)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks expected feature columns for {dataset}: {missing}")
    for col in ("subject", "y"):
        if col not in df.columns:
            raise ValueError(f"{path} lacks required id column {col!r}")
    return df


def load_locked_feature_matrix(dataset: str) -> pd.DataFrame:
    """Load the exact matrix the locked classical floor consumed."""
    floor = load_floor_json(dataset)
    return load_feature_matrix(REPO_ROOT / floor["features_parquet"], dataset)


def assert_matches_locked_floor(dataset: str, parquet_path: Path | str | None = None) -> str:
    """Return the parquet SHA-256 after asserting it equals the locked floor hash."""
    floor = load_floor_json(dataset)
    path = Path(parquet_path) if parquet_path is not None else REPO_ROOT / floor["features_parquet"]
    digest = sha256_file(path)
    expected = floor["features_parquet_sha256"]
    if digest != expected:
        raise AssertionError(
            f"feature matrix for {dataset} is not byte-identical to the locked floor: "
            f"{path} sha256={digest} vs floor.json {expected}"
        )
    feat_cols = frozen.feature_columns(dataset)
    if floor["results"]["feature_columns"] != feat_cols:
        raise AssertionError(
            f"locked floor feature_columns for {dataset} drifted from frozen.feature_columns"
        )
    return digest
