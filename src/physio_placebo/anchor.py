"""Health-LLM fidelity-anchor helpers (PMData stress). Owned code, not a vendor copy.

Prompt construction matches the released ``inference.py`` zero-shot branch line-for-line
(instruction string, format-prompt sampling, ignored ``instruction`` field on the eval
item). Scoring is ours: Health-LLM shipped no evaluation code.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import sha256_file

SPLIT_MANIFEST = REPO_ROOT / "results" / "anchor" / "split_manifest.json"
DEFAULT_EVAL_JSON = REPO_ROOT / "data/third_party/Health-LLM/zero-shot/data/pmdata/stress.json"
SEEDS = (0, 1, 2)
STRESS_FORMAT_CHOICES = [0, 1, 2, 3, 4, 5]
ZERO_SHOT_INSTRUCTION = (
    "You are a health assistant. Your mission is to read the following input health "
    "query and return your prediction.\n"
)
PAPER_MEDALPACA = {
    "mae_mean": 0.76,
    "mae_sd": 0.10,
    "source": "Health-LLM (CHIL 2024) Table 3, PMData stress, zero-shot MedAlpaca-7b",
}
PAPER_GPT35_FEWSHOT = {
    "mae_mean": 0.94,
    "mae_sd": 0.10,
    "source": "Health-LLM (CHIL 2024) Table 3, PMData stress, few-shot GPT-3.5",
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_LABEL = re.compile(r"is (\d+)")


def locked_eval_sha256() -> str:
    return json.loads(SPLIT_MANIFEST.read_text())["adapted_eval_sha256"]


def load_eval_items(path: Path | None = None) -> list[dict[str, Any]]:
    eval_path = path or DEFAULT_EVAL_JSON
    digest = sha256_file(eval_path)
    expected = locked_eval_sha256()
    if digest != expected:
        raise ValueError(
            f"eval split hash mismatch: {eval_path} sha256={digest} expected={expected}"
        )
    items = json.loads(eval_path.read_text())
    if len(items) != 299:
        raise ValueError(f"expected 299 eval items, got {len(items)}")
    return items


def parse_first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    return float(match.group(0)) if match else None


def parse_label(label_sentence: str) -> int:
    match = _LABEL.search(label_sentence)
    if match is None:
        raise ValueError(f"unparsable label: {label_sentence!r}")
    return int(match.group(1))


def set_seed(seed: int) -> None:
    """Match Health-LLM ``set_seed`` (defined, not called, in the released script)."""
    random.seed(seed)
    np.random.seed(seed)


def format_prompt(rand_num: int | float | str) -> str:
    # Same template as released inference.py (their `.format(rand_num)`).
    return f"For example, the answer should be in the following format:\nAnswer: {rand_num}"


def zero_shot_question(item_input: str, rand_num: int | float | str) -> str:
    return ZERO_SHOT_INSTRUCTION + item_input + format_prompt(rand_num)


def build_zero_shot_prompts(
    items: Sequence[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Reconstruct the released zero-shot loop for one seed.

    Their eval JSON carries an ``instruction`` field; the zero-shot branch ignores it and
    prefixes ``ZERO_SHOT_INSTRUCTION`` instead. Format-prompt numbers are sampled from
    ``{0..5}`` even though the declared stress range is 1–5.
    """
    set_seed(seed)
    prompts = []
    for i, item in enumerate(items):
        rand_num = random.choice(STRESS_FORMAT_CHOICES)
        prompts.append(
            {
                "no": i + 1,
                "rand_num": rand_num,
                "question": zero_shot_question(item["input"], rand_num),
                "label": item["output"],
            }
        )
    return prompts


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    abs_errors: list[float] = []
    failures = 0
    for rec in records:
        label = parse_label(rec["label"])
        pred = parse_first_number(str(rec.get("answer", "")))
        if pred is None:
            failures += 1
            continue
        abs_errors.append(abs(pred - label))
    if not abs_errors:
        raise ValueError("no parsable predictions")
    return {
        "n_items": len(records),
        "n_parsed": len(abs_errors),
        "n_parse_failures": failures,
        "mae": statistics.mean(abs_errors),
        "parse_failure_rate": failures / len(records),
    }


def constant3_mae(items: Sequence[dict[str, Any]]) -> float:
    labels = [parse_label(item["output"]) for item in items]
    return statistics.mean(abs(3 - y) for y in labels)


def label_distribution(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    labels = [parse_label(item["output"]) for item in items]
    return {str(k): v for k, v in sorted(Counter(labels).items())}


def aggregate_seed_maes(per_seed: dict[int, dict[str, Any]]) -> tuple[float, float]:
    maes = [per_seed[s]["mae"] for s in SEEDS]
    return statistics.mean(maes), statistics.stdev(maes)


def remaining_prompts(
    prompts: Sequence[dict[str, Any]], done: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return prompts not yet in a checkpoint. Partials must be a prefix of ``no``."""
    n_done = len(done)
    if n_done > len(prompts):
        raise ValueError(f"partial has {n_done} records but only {len(prompts)} prompts")
    if n_done and int(done[-1]["no"]) != n_done:
        raise ValueError(
            f"partial is not a contiguous prefix: last no={done[-1]['no']} len={n_done}"
        )
    return list(prompts[n_done:])

