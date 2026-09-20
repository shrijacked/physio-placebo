"""Freeze winning prompt IDs from real vLLM prompt-dev sweeps. Not from ConstantClient."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from physio_placebo.paths import REPO_ROOT, configs_dir
from physio_placebo.prompts.sweep import load_sweep_spec, pick_winner
from physio_placebo.provenance import git_sha, utc_stamp


class FreezeError(ValueError):
    pass


def lock_path() -> Path:
    return configs_dir() / "frozen_prompts.lock.yaml"


def lock_path_c() -> Path:
    return configs_dir() / "frozen_prompts_c.lock.yaml"


def _rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_sweep(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def assert_real_vllm_sweep(payload: dict[str, Any]) -> None:
    client = (payload.get("config") or {}).get("client")
    if client != "vllm":
        raise FreezeError(f"refuse to freeze from client={client!r}; need vllm")
    cells = payload.get("cells") or []
    if not cells:
        raise FreezeError("sweep has no cells")
    if any(c.get("macro_f1") != c.get("macro_f1") for c in cells):
        raise FreezeError("sweep has NaN macro_f1 cells")


def _group_cells(cells: Sequence[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for r in cells:
        key = (r["dataset"], r["model"], r["paradigm"], r["scheme"])
        grouped.setdefault(key, []).append(r)
    return grouped


def slim_winner(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "model": row["model"],
        "paradigm": row["paradigm"],
        "scheme": row["scheme"],
        "template_id": row["template_id"],
        "n_shot": int(row["n_shot"]),
        "macro_f1": float(row["macro_f1"]),
        "n_scored": int(row["n_scored"]),
        "n_failures": int(row["n_failures"]),
        "prompt_dev_subjects": list(row["prompt_dev_subjects"]),
    }


def winners_from_sweep(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [slim_winner(pick_winner(rows)) for rows in _group_cells(payload["cells"]).values()]


def build_lockfile(paths: Sequence[Path]) -> dict[str, Any]:
    spec = load_sweep_spec()
    winners: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in paths:
        path = Path(path)
        payload = load_sweep(path)
        assert_real_vllm_sweep(payload)
        winners.extend(winners_from_sweep(payload))
        sources.append(_rel(path))
    if any(w["paradigm"] == "C" for w in winners):
        raise FreezeError("A/B lockfile cannot contain Paradigm C; use frozen_prompts_c.lock.yaml")
    keys = [(w["dataset"], w["model"], w["paradigm"], w["scheme"]) for w in winners]
    if len(keys) != len(set(keys)):
        raise FreezeError("duplicate dataset/model/paradigm/scheme in lockfile")
    n_expected = len(spec["datasets"]) * len(spec["paradigms"]) * 2  # two models
    if len(winners) != n_expected:
        raise FreezeError(f"expected {n_expected} frozen cells, got {len(winners)}")
    return {
        "frozen": True,
        "frozen_at_utc": utc_stamp(),
        "git_sha_at_freeze": git_sha(),
        "seed": int(spec["seed"]),
        "n_dev": int(spec["n_dev"]),
        "source_sweeps": sources,
        "winners": winners,
        "note": "Prompt IDs frozen from real vLLM prompt-dev sweeps. Do not retune.",
    }


def write_lockfile(paths: Sequence[Path], dest: Path | None = None) -> Path:
    dest = Path(dest) if dest is not None else lock_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(build_lockfile(paths), sort_keys=False))
    return dest


def build_lockfile_c(paths: Sequence[Path]) -> dict[str, Any]:
    spec = load_sweep_spec()
    winners: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in paths:
        path = Path(path)
        payload = load_sweep(path)
        assert_real_vllm_sweep(payload)
        winners.extend(winners_from_sweep(payload))
        sources.append(_rel(path))
    if not winners:
        raise FreezeError("C sweep has no winners")
    for w in winners:
        if w["paradigm"] != "C":
            raise FreezeError("C lockfile may only contain paradigm C")
        if w["dataset"] not in {"wesad", "maus"}:
            raise FreezeError(f"Paradigm C is WESAD and MAUS only, not {w['dataset']}")
        if w["scheme"] is not None:
            raise FreezeError("Paradigm C has no downsample scheme")
    keys = [(w["dataset"], w["model"], w["paradigm"], w["scheme"]) for w in winners]
    if len(keys) != len(set(keys)):
        raise FreezeError("duplicate dataset/model/paradigm/scheme in C lockfile")
    models = {w["model"] for w in winners}
    n_expected = 2 * len(models)
    if len(winners) != n_expected:
        raise FreezeError(f"expected {n_expected} C cells, got {len(winners)}")
    return {
        "frozen": True,
        "frozen_at_utc": utc_stamp(),
        "git_sha_at_freeze": git_sha(),
        "seed": int(spec["seed"]),
        "n_dev": int(spec["n_dev"]),
        "source_sweeps": sources,
        "winners": winners,
        "note": "Paradigm C prompt IDs frozen from a real vLLM VL sweep. Do not retune A/B.",
    }


def write_lockfile_c(paths: Sequence[Path], dest: Path | None = None) -> Path:
    dest = Path(dest) if dest is not None else lock_path_c()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(build_lockfile_c(paths), sort_keys=False))
    return dest


def load_frozen_prompts_c() -> dict[str, Any]:
    path = lock_path_c()
    if not path.exists():
        raise FreezeError(f"C prompts not frozen: missing {path}")
    payload = yaml.safe_load(path.read_text())
    if not payload or not payload.get("frozen"):
        raise FreezeError("C lockfile exists but frozen is not true")
    return payload


def load_frozen_prompts() -> dict[str, Any]:
    path = lock_path()
    if not path.exists():
        raise FreezeError(f"prompts not frozen: missing {path}")
    payload = yaml.safe_load(path.read_text())
    if not payload or not payload.get("frozen"):
        raise FreezeError("lockfile exists but frozen is not true")
    return payload


def frozen_choice(dataset: str, model: str, paradigm: str, scheme: str | None) -> dict[str, Any]:
    locks = [load_frozen_prompts()]
    if paradigm == "C":
        locks = [load_frozen_prompts_c()]
    for lock in locks:
        for w in lock["winners"]:
            if (
                w["dataset"] == dataset
                and w["model"] == model
                and w["paradigm"] == paradigm
                and w["scheme"] == scheme
            ):
                return w
    raise KeyError(f"no frozen prompt for {dataset}/{model}/{paradigm}/{scheme}")
