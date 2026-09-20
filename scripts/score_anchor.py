"""Score a Health-LLM fidelity-anchor run (PMData stress).

Health-LLM released no evaluation code, so scoring follows the convention in
docs/fidelity-anchor.md: the prediction is the FIRST number in the response;
unparsable items are dropped from MAE and counted as parse failures.

Default paths still point at the abandoned gpt-3.5 few-shot cell. Pass
``--pred-dir`` for the MedAlpaca zero-shot outputs.

Writes: results/anchor/metrics.json (overwritten; the timestamped run dir is the
canonical artifact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.anchor import (
    DEFAULT_EVAL_JSON,
    PAPER_GPT35_FEWSHOT,
    PAPER_MEDALPACA,
    SEEDS,
    aggregate_seed_maes,
    constant3_mae,
    label_distribution,
    load_eval_items,
    locked_eval_sha256,
    score_records,
)
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import git_sha, sha256_file, utc_stamp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-dir",
        default="data/third_party/Health-LLM/output/gpt-3.5/few-shot",
        help="directory containing stress_sd{0,1,2}.json",
    )
    ap.add_argument(
        "--paper",
        choices=["gpt35-fewshot", "medalpaca-zeroshot"],
        default="gpt35-fewshot",
    )
    args = ap.parse_args()

    paper = PAPER_MEDALPACA if args.paper == "medalpaca-zeroshot" else PAPER_GPT35_FEWSHOT
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = REPO_ROOT / pred_dir

    items = load_eval_items()
    per_seed = {}
    for seed in SEEDS:
        pred_file = pred_dir / f"stress_sd{seed}.json"
        records = json.loads(pred_file.read_text())
        scored = score_records(records)
        scored["pred_file_sha256"] = sha256_file(pred_file)
        per_seed[seed] = scored

    mae_mean, mae_sd = aggregate_seed_maes(per_seed)
    metrics = {
        "task": "PMData stress (Health-LLM eval split, seed-123 shuffle, 299 items)",
        "paper_target": paper,
        "scoring": "first number in response; parse failures dropped from MAE and counted",
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "mae_mean": round(mae_mean, 4),
        "mae_sd": round(mae_sd, 4),
        "constant3_baseline_mae": round(constant3_mae(items), 4),
        "gap_vs_paper": round(mae_mean - paper["mae_mean"], 4),
        "eval_n": len(items),
        "eval_label_distribution": label_distribution(items),
        "eval_file": str(DEFAULT_EVAL_JSON.relative_to(REPO_ROOT)),
        "eval_file_sha256": locked_eval_sha256(),
        "git_sha": git_sha(),
        "scored_at_utc": utc_stamp(),
    }

    dest = REPO_ROOT / "results/anchor/metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"eval items: {metrics['eval_n']}, label dist: {metrics['eval_label_distribution']}")
    for seed in SEEDS:
        s = per_seed[seed]
        print(
            f"seed {seed}: MAE={s['mae']:.4f}  parsed={s['n_parsed']}/{s['n_items']}"
            f"  failures={s['n_parse_failures']}"
        )
    print(f"MAE mean +/- sd: {mae_mean:.3f} +/- {mae_sd:.3f}")
    print(f"constant-3 baseline MAE: {metrics['constant3_baseline_mae']}")
    print(
        f"paper target: {paper['mae_mean']} +/- {paper['mae_sd']}"
        f"  ->  gap: {mae_mean - paper['mae_mean']:+.3f}"
    )
    print(f"wrote {dest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
