"""Health-LLM comparison arm. Lock file plus train-split writer.

The WESAD / MAUS / CogWear grid is not read or written here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from physio_placebo.anchor import (
    DEFAULT_EVAL_JSON,
    DEFAULT_TRAIN_POOL_JSON,
    assert_disjoint,
    load_eval_items,
    train_complement,
)
from physio_placebo.paths import configs_dir
from physio_placebo.provenance import config_hash, sha256_bytes, sha256_file

LOCK_PATH = configs_dir() / "healthllm_finetune.lock.yaml"
REQUIRED_LOCK_KEYS = (
    "model",
    "use_lora",
    "num_epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "global_batch_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "model_max_length",
    "warmup_steps",
    "seed",
    "eval_seed",
    "max_new_tokens",
    "do_sample",
)


def load_finetune_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = path or LOCK_PATH
    data = yaml.safe_load(lock_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"lock file is not a mapping: {lock_path}")
    missing = [key for key in REQUIRED_LOCK_KEYS if key not in data]
    if missing:
        raise ValueError(f"lock file missing keys: {missing}")
    if data["model"] != "medalpaca/medalpaca-7b":
        raise ValueError(f"unexpected model: {data['model']}")
    if data["use_lora"] is not True:
        raise ValueError("this cell is the LoRA arm")
    if int(data["num_epochs"]) != 3:
        raise ValueError("7B appendix cell is 3 epochs")
    if int(data["per_device_batch_size"]) * int(data["gradient_accumulation_steps"]) != int(
        data["global_batch_size"]
    ):
        raise ValueError("per-device batch times accumulation must equal global batch")
    return data


def load_train_pool(path: Path | None = None) -> list[dict[str, Any]]:
    pool_path = path or DEFAULT_TRAIN_POOL_JSON
    items = json.loads(pool_path.read_text())
    if not isinstance(items, list) or not items:
        raise ValueError(f"train pool is empty: {pool_path}")
    return items


def build_locked_train_items(
    pool: list[dict[str, Any]] | None = None,
    eval_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    train_pool = pool if pool is not None else load_train_pool()
    held_out = eval_items if eval_items is not None else load_eval_items()
    kept = train_complement(train_pool, held_out)
    assert_disjoint(kept, held_out)
    if not kept:
        raise ValueError("train complement is empty")
    return kept


def train_split_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Manifest fields that do not depend on the clock."""
    body = json.dumps(items, indent=2) + "\n"
    return {
        "n_train": len(items),
        "train_sha256": sha256_bytes(body.encode()),
        "eval_file": str(DEFAULT_EVAL_JSON),
        "eval_sha256": sha256_file(DEFAULT_EVAL_JSON),
        "train_pool_file": str(DEFAULT_TRAIN_POOL_JSON),
        "train_pool_sha256": sha256_file(DEFAULT_TRAIN_POOL_JSON),
        "n_train_pool": len(load_train_pool()),
        "n_dropped": len(load_train_pool()) - len(items),
        "lock_sha256": sha256_file(LOCK_PATH),
        "lock_config_sha256": config_hash(load_finetune_lock()),
    }
