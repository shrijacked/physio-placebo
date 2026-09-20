"""Prompt freeze: only real vLLM sweeps, then a lockfile the grid can read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.freeze import (
    FreezeError,
    build_lockfile,
    frozen_choice,
    load_frozen_prompts,
    write_lockfile,
)
from physio_placebo.prompts.sweep import pick_winner

QWEN = REPO_ROOT / "results" / "prompt_sweep" / "sweep_vllm_qwen3_8b.json"
LLAMA = REPO_ROOT / "results" / "prompt_sweep" / "sweep_vllm_llama31_8b.json"


def test_refuse_constant_client_sweep(tmp_path: Path):
    fake = {
        "config": {"client": "constant", "model": "constant"},
        "cells": [
            {
                "dataset": "maus",
                "model": "constant",
                "paradigm": "B",
                "scheme": None,
                "template_id": "instruction",
                "n_shot": 0,
                "macro_f1": 0.5,
                "n_scored": 1,
                "n_failures": 0,
                "prompt_dev_subjects": ["015", "023"],
            }
        ],
        "winners": [],
        "frozen": False,
    }
    path = tmp_path / "sweep_constant_constant.json"
    path.write_text(json.dumps(fake))
    with pytest.raises(FreezeError, match="vllm"):
        build_lockfile([path])


def test_real_vllm_sweeps_freeze_to_18_winners_matching_pick_winner():
    lock = build_lockfile([QWEN, LLAMA])
    assert lock["frozen"] is True
    assert lock["seed"] == 1337
    assert len(lock["winners"]) == 18
    keys = {
        (w["dataset"], w["model"], w["paradigm"], w["scheme"])
        for w in lock["winners"]
    }
    assert len(keys) == 18
    for path in (QWEN, LLAMA):
        payload = json.loads(path.read_text())
        grouped: dict[tuple, list] = {}
        for r in payload["cells"]:
            grouped.setdefault(
                (r["dataset"], r["model"], r["paradigm"], r["scheme"]), []
            ).append(r)
        for key, rows in grouped.items():
            want = pick_winner(rows)
            got = next(
                w
                for w in lock["winners"]
                if (w["dataset"], w["model"], w["paradigm"], w["scheme"]) == key
            )
            assert got["template_id"] == want["template_id"]
            assert got["n_shot"] == want["n_shot"]


def test_write_lockfile_and_frozen_choice(tmp_path: Path, monkeypatch):
    from physio_placebo.prompts import freeze as freeze_mod

    dest = tmp_path / "frozen_prompts.lock.yaml"
    monkeypatch.setattr(freeze_mod, "lock_path", lambda: dest)
    write_lockfile([QWEN, LLAMA], dest=dest)
    choice = frozen_choice("wesad", "qwen3_8b", "A", "uniform_stride")
    assert choice["template_id"] == "instruction"
    assert choice["n_shot"] == 0


def test_committed_lockfile_is_frozen():
    lock = load_frozen_prompts()
    assert lock["frozen"] is True
    assert len(lock["winners"]) == 18
    assert frozen_choice("cogwear", "llama31_8b", "B", None)["template_id"] == "minimal"
