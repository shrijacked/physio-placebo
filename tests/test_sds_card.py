"""A gated SDS cell is words, not a ratio."""

from __future__ import annotations

import json

from physio_placebo.report.card import build_card, render_markdown


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
