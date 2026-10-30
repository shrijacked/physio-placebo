"""A gated SDS cell is words, not a ratio."""

from __future__ import annotations

import json

import numpy as np
import pytest

from physio_placebo.report import card as card_mod
from physio_placebo.report.card import build_card, render_markdown
from physio_placebo.stats.secondary import matched_permutation_floor


def test_gated_cell_prints_undefined_and_not_the_ungated_ratio(tmp_path):
    root = tmp_path
    floor = {
        "results": {
            "models": {"logreg_l2": {"pooled_macro_f1": 0.90}},
            "chance": {"permutation": {"logreg_l2": {"mean": 0.42}}},
        }
    }
    floor_path = root / "results" / "classical_floor" / "fake"
    floor_path.mkdir(parents=True)
    (floor_path / "floor.json").write_text(json.dumps(floor))
    (root / "configs").mkdir()
    (root / "configs" / "locked_floors.yaml").write_text("wesad: results/classical_floor/fake/floor.json\n")
    # (0.50 - 0.45) / (0.50 - 0.42) = 0.625. The gate must hide that number.
    real = {
        "cells": [
            {
                "dataset": "wesad",
                "model": "qwen3_8b",
                "paradigm": "A",
                "scheme": "bin_mean",
                "macro_f1": 0.50,
                "ci_lo": 0.40,
                "ci_hi": 0.55,
                "per_subject": {"s1": 0.5, "s2": 0.5},
                "eval_subjects": ["s1", "s2"],
            }
        ]
    }
    grid = root / "results" / "eval_grid"
    grid.mkdir(parents=True)
    (grid / "eval_vllm_qwen3_8b.json").write_text(json.dumps(real))
    surr_dir = root / "results" / "surrogates" / "eval" / "qwen3_8b" / "wesad" / "A" / "bin_mean" / "gaussian"
    surr_dir.mkdir(parents=True)
    (surr_dir / "cell.json").write_text(
        json.dumps(
            {
                "cell": {
                    "macro_f1": 0.45,
                    "per_subject": {"s1": 0.4, "s2": 0.5},
                    "rows": [],
                }
            }
        )
    )
    card = build_card(root)
    text = render_markdown(card)
    assert "undefined (validity gate)" in text
    assert "0.625" not in text
    assert card["primary"][0]["sds"] == "undefined (validity gate)"
    assert "pending real rows" in text


def test_real_side_cache_does_not_reuse_another_setup_when_ids_collide(monkeypatch):
    """Two setups that share an object id must not share a chance floor."""
    monkeypatch.setattr(card_mod, "id", lambda _obj: 0, raising=False)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    subjects = np.array(["s0"] * 10)

    def pack(predictions):
        return {
            "y": labels,
            "yhat": np.array(predictions),
            "subjects": subjects,
            "margin": np.zeros(10),
            "auroc": float("nan"),
        }

    always_a = pack([0] * 10)
    mixed = pack([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    cache: dict = {}
    wesad = ("qwen3_8b", "wesad", "A", "bin_mean")
    maus = ("qwen3_8b", "maus", "A", "bin_mean")
    first = card_mod._real_side(wesad, always_a, 1.0, cache)
    second = card_mod._real_side(maus, mixed, 1.0, cache)
    own = matched_permutation_floor(subjects, labels, mixed["yhat"])
    assert second["floor"]["mean"] == pytest.approx(own["mean"])
    assert second["floor"]["mean"] != pytest.approx(first["floor"]["mean"])
