"""Real-signal eval grid: frozen prompts, eval subjects only, no GPU."""

from __future__ import annotations

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.prompts.eval_grid import run_eval_cell
from physio_placebo.prompts.freeze import frozen_choice
from physio_placebo.scoring.client import ConstantClient


def test_maus_b_eval_excludes_prompt_dev_and_emits_ci():
    rows = run_eval_cell(
        "maus",
        ConstantClient(),
        model_key="qwen3_8b",
        paradigm="B",
        scheme=None,
    )
    subjects = load_locked_feature_matrix("maus")["subject"].astype(str).unique()
    dev, eval_ = prompt_dev_split(sorted(subjects), n_dev=2, seed=1337)
    assert set(rows["prompt_dev_subjects"]) == set(dev)
    assert set(rows["eval_subjects"]) == set(eval_)
    assert set(rows["scored_subjects"]).isdisjoint(dev)
    assert set(rows["scored_subjects"]) == set(eval_)
    choice = frozen_choice("maus", "qwen3_8b", "B", None)
    assert rows["template_id"] == choice["template_id"]
    assert rows["n_shot"] == choice["n_shot"]
    assert rows["n_scored"] > 0
    assert rows["n_failures"] == 0
    assert rows["ci_lo"] <= rows["macro_f1"] <= rows["ci_hi"]
