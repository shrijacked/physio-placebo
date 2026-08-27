"""Freeze guard: after Week-1 freeze, any edit to frozen.py fails this test."""

from __future__ import annotations

import json

import pytest

from physio_placebo.features.frozen import frozen_code_hash
from physio_placebo.paths import configs_dir

LOCK = configs_dir() / "frozen_features.lock.json"


@pytest.mark.skipif(not LOCK.exists(), reason="feature code not frozen yet (pre-Week-1 freeze)")
def test_frozen_feature_code_unchanged():
    lock = json.loads(LOCK.read_text())
    current = frozen_code_hash()
    assert current["frozen_py_sha256"] == lock["frozen_py_sha256"], (
        "src/physio_placebo/features/frozen.py changed after the Week-1 freeze. "
        "This requires explicit approval; if approved, rerun scripts/freeze_features.py "
        "and regenerate every feature matrix."
    )
    assert current["neurokit2"] == lock["neurokit2"]
