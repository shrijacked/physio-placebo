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


def resolve_healthllm_root() -> Path:
    """Released clone lives under ``configs/data/...`` in this checkout.

    ``data/third_party/Health-LLM`` is checked first so a future symlink still wins.
    """
    candidates = (
        REPO_ROOT / "data" / "third_party" / "Health-LLM",
        REPO_ROOT / "configs" / "data" / "third_party" / "Health-LLM",
    )
    marker = Path("zero-shot") / "data" / "pmdata" / "stress.json"
    for root in candidates:
        if (root / marker).is_file():
            return root
    return candidates[-1]


HEALTHLLM_ROOT = resolve_healthllm_root()
DEFAULT_EVAL_JSON = HEALTHLLM_ROOT / "zero-shot" / "data" / "pmdata" / "stress.json"
DEFAULT_TRAIN_POOL_JSON = HEALTHLLM_ROOT / "PMData_stress_train_all.json"
MEDALPACA_PROMPT = {
    "primer": (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the request."
    ),
    "instruction": "\n\n### Instruction:\n",
    "input": "\n\n### Input:\n",
    "output": "\n\n### Response:\n",
}
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
_PAPER_LISTS = {
    "steps": re.compile(r"\[Steps\]: \[([^\]]*)\]"),
    "calories": re.compile(r"\[Burned Calorories\]: \[([^\]]*)\]"),
    "rhr": re.compile(r"\[Resting Heart Rate\]: \[([^\]]*)\]"),
    "sleep": re.compile(r"\[SleepMinutes\]: \[([^\]]*)\]"),
}
_PAPER_MOOD = re.compile(r"\[Mood\]: (\d+(?:\.\d+)?) out of 5")
# Figure 7, zero-shot exemplar. Table 1 is the basic (averaged) context.
PAPER_BASIC_INSTRUCTION = "You are an intelligent healthcare agent."


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


def medalpaca_template() -> dict[str, str]:
    """Vendored ``medalpaca/prompt_templates/medalpaca.json``. Not rewritten."""
    rel = Path("medalpaca") / "prompt_templates" / "medalpaca.json"
    roots = (
        resolve_healthllm_root(),
        REPO_ROOT / "data" / "third_party" / "Health-LLM",
        REPO_ROOT / "configs" / "data" / "third_party" / "Health-LLM",
    )
    for root in roots:
        path = root / rel
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(rel)


def medalpaca_generate_prompt(
    instruction: str | None = None,
    input: str | None = None,
    output: str | None = None,
) -> str:
    """Copy of ``DataHandler.generate_prompt`` in their ``handler.py``.

    The f-string is theirs. ``instruction``, ``input``, and ``output`` are the
    same arguments ``Inferer.__call__`` passes through.
    """
    template = medalpaca_template()
    if not any([instruction, input, output]):
        raise ValueError("At least one of `instruction`, `input`, `output` should be defined")
    return (
        f'{template["primer"]}'
        f'{template["instruction"]}{instruction or ""}'
        f'{template["input"]}{input or ""}'
        f'{template["output"]}{output or ""}'
    )


def medalpaca_wrap_released_question(question: str) -> str:
    """Split a released zero-shot string into the two slots ``generate_prompt`` has."""
    if question.startswith(ZERO_SHOT_INSTRUCTION):
        return medalpaca_generate_prompt(
            instruction=ZERO_SHOT_INSTRUCTION.strip(),
            input=question[len(ZERO_SHOT_INSTRUCTION) :],
        )
    return medalpaca_generate_prompt(input=question)


def _mean_bracket_list(text: str, kind: str) -> float:
    match = _PAPER_LISTS[kind].search(text)
    if match is None:
        raise ValueError(f"no {kind} list in sensor text")
    values = [float(part) for part in match.group(1).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty {kind} list")
    return sum(values) / len(values)


def paper_basic_question(item_input: str) -> str:
    """Table 1 basic context, with the stress range from their generator.

    ``gen_dataset.py`` already averages the 14-day window (``steps_14d`` and
    the three siblings) and then writes the raw lists into the string. Table 1
    is the paragraph of those averages. Stress is range 1 to 5 in that script.
    """
    steps = _mean_bracket_list(item_input, "steps")
    calories = _mean_bracket_list(item_input, "calories")
    rhr = _mean_bracket_list(item_input, "rhr")
    sleep = _mean_bracket_list(item_input, "sleep")
    mood_match = _PAPER_MOOD.search(item_input)
    if mood_match is None:
        raise ValueError("no mood value in sensor text")
    mood = float(mood_match.group(1))
    return (
        "The analysis of recent period: 14 days averaged sensor readings show: "
        f"Steps: {steps:.1f} steps, Burned Calories: {calories:.1f} calories, "
        f"Resting Heart Rate: {rhr:.2f} beats/min, Sleep Minutes: {sleep:.1f} minutes, "
        f"Mood: {mood:g} out of 5. In this regard, what is the predicted stress "
        "score/level between 1 and 5?"
    )


def paper_basic_prompt(item_input: str) -> str:
    """Figure 7 zero-shot frame around the Table 1 basic paragraph."""
    return (
        f"Instruction: {PAPER_BASIC_INSTRUCTION}\n"
        f"Question: {paper_basic_question(item_input)}\n"
        "Response:"
    )


_LIST_PREC = {"steps": ".1f", "calories": ".1f", "rhr": ".2f", "sleep": ".1f"}


def _bracket_list_text(text: str, kind: str) -> str:
    """Figure 7 prints heart rate as ``65.61`` and the other sensors as ``5260.0``."""
    match = _PAPER_LISTS[kind].search(text)
    if match is None:
        raise ValueError(f"no {kind} list in sensor text")
    values = [float(part) for part in match.group(1).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty {kind} list")
    rendered = ", ".join(format(value, _LIST_PREC[kind]) for value in values)
    return "[" + rendered + "]"


def paper_figure7_question(item_input: str) -> str:
    """Figure 7 zero-shot question: the days as bullets, then their question line.

    The printed exemplar is readiness. Stress uses the same sentence with the
    range from ``gen_dataset.py`` (1 to 5). The age/sex/height line in the
    figure is not in the released stress questions, and their participant table
    is only a stub, so it is left out.
    """
    mood_match = _PAPER_MOOD.search(item_input)
    if mood_match is None:
        raise ValueError("no mood value in sensor text")
    mood = float(mood_match.group(1))
    return (
        "The analysis of recent 2-weeks sensor readings show:\n"
        f"  • Steps: {_bracket_list_text(item_input, 'steps')} steps\n"
        f"  • Burned Calories: {_bracket_list_text(item_input, 'calories')} calories\n"
        f"  • Resting Heart Rate: {_bracket_list_text(item_input, 'rhr')} beats/min\n"
        f"  • Sleep Minutes: {_bracket_list_text(item_input, 'sleep')} minutes\n"
        f"  • Mood: {mood:g} out of 5\n"
        "In this regard, can you predict user's stress level between 1 and 5?"
    )


def paper_figure7_list_prompt(item_input: str) -> str:
    """Figure 7 frame around the bullet-list question."""
    return (
        f"Instruction: {PAPER_BASIC_INSTRUCTION}\n"
        f"Question: {paper_figure7_question(item_input)}\n"
        "Response:"
    )


def paper_figure1_prompt(item_input: str) -> str:
    """Figure 1 frame: ``### Instruction`` / ``### Question`` / ``### Response``."""
    return (
        f"### Instruction: {PAPER_BASIC_INSTRUCTION}\n"
        f"### Question: {paper_basic_question(item_input)}\n"
        "### Response:"
    )


def medalpaca_wrapped_question(item_input: str, rand_num: int | float | str) -> str:
    """Their ``generate_prompt`` with the two pieces the zero-shot loop concatenates.

    Instruction is the released health-assistant line. Input is the sensor text
    plus their ``Answer:`` example. ``Inferer`` passes those as separate arguments.
    """
    return medalpaca_generate_prompt(
        instruction=ZERO_SHOT_INSTRUCTION.strip(),
        input=item_input + format_prompt(rand_num),
    )


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
    maes = [per_seed[s]["mae"] for s in sorted(per_seed)]
    if len(maes) < 2:
        return maes[0], 0.0
    return statistics.mean(maes), statistics.stdev(maes)


def record_body(item: dict[str, Any]) -> str:
    """Sensor text. Eval JSON uses ``input``; their generator's eval writer used ``question``."""
    body = item.get("input", item.get("question"))
    if not isinstance(body, str) or body == "":
        raise ValueError("record has no input or question text")
    return body


def train_complement(
    pool: Sequence[dict[str, Any]], eval_items: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Train rows whose sensor text is not in the locked eval split."""
    banned = {record_body(item) for item in eval_items}
    return [item for item in pool if record_body(item) not in banned]


def assert_disjoint(
    train_items: Sequence[dict[str, Any]], eval_items: Sequence[dict[str, Any]]
) -> None:
    overlap = {record_body(item) for item in train_items} & {
        record_body(item) for item in eval_items
    }
    if overlap:
        raise ValueError(f"eval text leaked into train: {len(overlap)} items")


def medalpaca_train_text(item: dict[str, Any]) -> str:
    """MedAlpaca default template (``prompt_templates/medalpaca.json``)."""
    instruction = item.get("instruction")
    sensor = item.get("input")
    output = item.get("output")
    if not isinstance(instruction, str) or not isinstance(sensor, str) or not isinstance(output, str):
        raise ValueError("train item needs instruction, input, and output strings")
    parts = MEDALPACA_PROMPT
    return (
        parts["primer"]
        + parts["instruction"]
        + instruction
        + parts["input"]
        + sensor
        + parts["output"]
        + output
    )


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

