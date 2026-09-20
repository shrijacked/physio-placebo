"""Paradigm C sweep / freeze / eval. ConstantClient only — no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physio_placebo.data.loso import prompt_dev_split
from physio_placebo.features.matrix import load_locked_feature_matrix
from physio_placebo.paradigms.series import load_windows
from physio_placebo.prompts.eval_grid import run_eval_cell
from physio_placebo.prompts.freeze import FreezeError, build_lockfile, build_lockfile_c, write_lockfile_c
from physio_placebo.prompts.sweep import run_dataset_sweep
from physio_placebo.scoring.client import ConstantClient

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


def _fake_plot_body(sw, *, t0, seg_name, dataset):
    return "The recording is the attached line plot.", _FAKE_PNG


def test_cogwear_c_sweep_is_rejected():
    with pytest.raises(ValueError, match="MAUS"):
        run_dataset_sweep("cogwear", ConstantClient(), paradigm="C", scheme=None, model_key="constant")


def test_wesad_c_sweep_loads_only_prompt_dev_windows(monkeypatch):
    loaded: list[str] = []
    real = load_windows

    def spy(dataset: str, subject: str):
        loaded.append(str(subject))
        return real(dataset, subject)

    monkeypatch.setattr("physio_placebo.prompts.sweep.load_windows", spy)
    monkeypatch.setattr("physio_placebo.prompts.sweep.plot_body", _fake_plot_body)
    rows = run_dataset_sweep(
        "wesad",
        ConstantClient(),
        paradigm="C",
        scheme=None,
        model_key="constant",
    )
    subjects = load_locked_feature_matrix("wesad")["subject"].astype(str).unique()
    dev, eval_ = prompt_dev_split(sorted(subjects), n_dev=2, seed=1337)
    assert set(loaded) == set(dev)
    assert set(eval_).isdisjoint(loaded)
    assert len(rows) == 6
    for r in rows:
        assert r["paradigm"] == "C"
        assert r["scheme"] is None
        assert r["n_scored"] > 0
        assert r["n_failures"] == 0


def test_ab_lockfile_refuses_paradigm_c(tmp_path: Path):
    fake = {
        "config": {"client": "vllm", "model": "qwen25_vl"},
        "cells": [
            {
                "dataset": "wesad",
                "model": "qwen25_vl",
                "paradigm": "C",
                "scheme": None,
                "template_id": "instruction",
                "n_shot": 0,
                "macro_f1": 0.5,
                "n_scored": 1,
                "n_failures": 0,
                "prompt_dev_subjects": ["S2", "S8"],
            }
        ],
    }
    path = tmp_path / "sweep_vllm_qwen25_vl.json"
    path.write_text(json.dumps(fake))
    with pytest.raises(FreezeError, match="Paradigm C"):
        build_lockfile([path])


def test_c_lockfile_from_vllm_sweep(tmp_path: Path):
    cells = []
    for dataset, dev in (("wesad", ["S2", "S8"]), ("maus", ["015", "023"])):
        for template_id, n_shot, f1 in (
            ("instruction", 0, 0.40),
            ("clinical", 0, 0.55),
            ("minimal", 4, 0.30),
        ):
            cells.append(
                {
                    "dataset": dataset,
                    "model": "qwen25_vl",
                    "paradigm": "C",
                    "scheme": None,
                    "template_id": template_id,
                    "n_shot": n_shot,
                    "macro_f1": f1,
                    "n_scored": 10,
                    "n_failures": 0,
                    "prompt_dev_subjects": dev,
                }
            )
    path = tmp_path / "sweep_vllm_qwen25_vl.json"
    path.write_text(json.dumps({"config": {"client": "vllm", "model": "qwen25_vl"}, "cells": cells}))
    lock = build_lockfile_c([path])
    assert lock["frozen"] is True
    assert len(lock["winners"]) == 2
    assert {w["dataset"] for w in lock["winners"]} == {"wesad", "maus"}
    assert all(w["template_id"] == "clinical" and w["n_shot"] == 0 for w in lock["winners"])


def test_wesad_c_eval_excludes_prompt_dev(tmp_path: Path, monkeypatch):
    from physio_placebo.prompts import freeze as freeze_mod

    cells = [
        {
            "dataset": "wesad",
            "model": "qwen25_vl",
            "paradigm": "C",
            "scheme": None,
            "template_id": "instruction",
            "n_shot": 0,
            "macro_f1": 0.5,
            "n_scored": 10,
            "n_failures": 0,
            "prompt_dev_subjects": ["S2", "S8"],
        },
        {
            "dataset": "maus",
            "model": "qwen25_vl",
            "paradigm": "C",
            "scheme": None,
            "template_id": "instruction",
            "n_shot": 0,
            "macro_f1": 0.5,
            "n_scored": 10,
            "n_failures": 0,
            "prompt_dev_subjects": ["015", "023"],
        },
    ]
    sweep = tmp_path / "sweep_vllm_qwen25_vl.json"
    sweep.write_text(json.dumps({"config": {"client": "vllm", "model": "qwen25_vl"}, "cells": cells}))
    dest = tmp_path / "frozen_prompts_c.lock.yaml"
    monkeypatch.setattr(freeze_mod, "lock_path_c", lambda: dest)
    monkeypatch.setattr("physio_placebo.prompts.sweep.plot_body", _fake_plot_body)
    write_lockfile_c([sweep], dest=dest)

    rows = run_eval_cell(
        "wesad",
        ConstantClient(),
        model_key="qwen25_vl",
        paradigm="C",
        scheme=None,
    )
    subjects = load_locked_feature_matrix("wesad")["subject"].astype(str).unique()
    dev, eval_ = prompt_dev_split(sorted(subjects), n_dev=2, seed=1337)
    assert set(rows["prompt_dev_subjects"]) == set(dev)
    assert set(rows["scored_subjects"]).isdisjoint(dev)
    assert set(rows["scored_subjects"]) == set(eval_)
    assert rows["template_id"] == "instruction"
    assert rows["n_shot"] == 0
    assert rows["ci_lo"] <= rows["macro_f1"] <= rows["ci_hi"]
