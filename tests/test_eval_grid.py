"""Real-signal eval grid: frozen prompts, eval subjects only, no GPU."""

from __future__ import annotations

import json
import subprocess
import sys

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.paths import REPO_ROOT
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


def test_save_rows_writes_logprobs_and_refuses_a_second_write(tmp_path):
    cmd = [
        sys.executable,
        "scripts/run_eval_grid.py",
        "--client",
        "constant",
        "--model",
        "qwen3_8b",
        "--dataset",
        "cogwear",
        "--paradigm",
        "B",
        "--save-rows",
        "--out-dir",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out = tmp_path / "eval_constant_qwen3_8b.json"
    payload = json.loads(out.read_text())
    assert payload["config"]["gpu_memory_utilization"] is None
    saved = payload["cells"][0]["rows"]
    assert saved
    assert set(saved[0]["logprobs"]) == {"A", "B"}
    first = out.read_bytes()
    again = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert again.returncode != 0
    assert "refusing to overwrite" in again.stderr + again.stdout
    assert out.read_bytes() == first
