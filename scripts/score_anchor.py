"""Score the Health-LLM fidelity-anchor reproduction (PMData stress, gpt-3.5-turbo-instruct).

Health-LLM released no evaluation code, so scoring follows the convention documented in
docs/fidelity-anchor.md before the run: the prediction is the FIRST number appearing in
the model's response text; responses with no parsable number are dropped from MAE and
reported as parse failures. MAE is computed per seed and aggregated as mean +/- sample sd
across seeds {0, 1, 2}. The constant-3 baseline is computed on the same eval items.

Reads:  data/third_party/Health-LLM/output/gpt-3.5/few-shot/stress_sd{0,1,2}.json
Writes: results/anchor/metrics.json
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter

from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import git_sha, sha256_file, utc_stamp

PAPER_TARGET = {
    "mae_mean": 0.94,
    "mae_sd": 0.10,
    "source": "Health-LLM (CHIL 2024) Table 3, PMData stress, few-shot GPT-3.5",
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_first_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    return float(match.group(0)) if match else None


def parse_label(label_sentence: str) -> int:
    match = re.search(r"is (\d+)", label_sentence)
    if match is None:
        raise ValueError(f"unparsable label: {label_sentence!r}")
    return int(match.group(1))


def main() -> None:
    root = REPO_ROOT
    out_dir = root / "data/third_party/Health-LLM/output/gpt-3.5/few-shot"
    eval_file = root / "data/third_party/Health-LLM/zero-shot/data/pmdata/stress.json"

    eval_items = json.loads(eval_file.read_text())
    eval_labels = [parse_label(item["output"]) for item in eval_items]

    per_seed = {}
    for seed in (0, 1, 2):
        pred_file = out_dir / f"stress_sd{seed}.json"
        records = json.loads(pred_file.read_text())
        abs_errors, failures = [], 0
        for rec in records:
            label = parse_label(rec["label"])
            pred = parse_first_number(rec["answer"])
            if pred is None:
                failures += 1
                continue
            abs_errors.append(abs(pred - label))
        per_seed[seed] = {
            "n_items": len(records),
            "n_parsed": len(abs_errors),
            "n_parse_failures": failures,
            "mae": statistics.mean(abs_errors),
            "pred_file_sha256": sha256_file(pred_file),
        }

    maes = [per_seed[s]["mae"] for s in (0, 1, 2)]
    mae_mean = statistics.mean(maes)
    mae_sd = statistics.stdev(maes)
    constant3_mae = statistics.mean(abs(3 - y) for y in eval_labels)

    metrics = {
        "task": "PMData stress (Health-LLM eval split, seed-123 shuffle, 299 items)",
        "model": "gpt-3.5-turbo-instruct, few-shot (their 3-exemplar prompt), max_tokens=120",
        "scoring": "first number in response; parse failures dropped from MAE and counted",
        "per_seed": per_seed,
        "mae_mean": round(mae_mean, 4),
        "mae_sd": round(mae_sd, 4),
        "constant3_baseline_mae": round(constant3_mae, 4),
        "paper_target": PAPER_TARGET,
        "gap_vs_paper": round(mae_mean - PAPER_TARGET["mae_mean"], 4),
        "eval_n": len(eval_labels),
        "eval_label_distribution": dict(sorted(Counter(eval_labels).items())),
        "eval_file_sha256": sha256_file(eval_file),
        "git_sha": git_sha(),
        "scored_at_utc": utc_stamp(),
    }

    dest = root / "results/anchor/metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"eval items: {len(eval_labels)}, label dist: {metrics['eval_label_distribution']}")
    for seed in (0, 1, 2):
        s = per_seed[seed]
        print(
            f"seed {seed}: MAE={s['mae']:.4f}  parsed={s['n_parsed']}/{s['n_items']}"
            f"  failures={s['n_parse_failures']}"
        )
    print(f"MAE mean +/- sd: {mae_mean:.3f} +/- {mae_sd:.3f}")
    print(f"constant-3 baseline MAE: {constant3_mae:.3f}")
    print(
        f"paper target: {PAPER_TARGET['mae_mean']} +/- {PAPER_TARGET['mae_sd']}"
        f"  ->  gap: {mae_mean - PAPER_TARGET['mae_mean']:+.3f}"
    )
    print(f"wrote {dest.relative_to(root)}")


if __name__ == "__main__":
    main()
