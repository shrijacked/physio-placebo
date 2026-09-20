"""Prompt-dev sweep. Does not freeze IDs and does not run the eval grid.

    python scripts/run_prompt_sweep.py --client constant --dataset maus --paradigm B
    python scripts/run_prompt_sweep.py --client vllm --model qwen3_8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.sweep import load_sweep_spec, pick_winner, run_dataset_sweep
from physio_placebo.provenance import run_metadata
from physio_placebo.scoring.client import ConstantClient


def _client(kind: str, model_key: str):
    if kind == "constant":
        return ConstantClient()
    if kind == "vllm":
        from physio_placebo.scoring.vllm_client import VLLMLogprobClient, prepare_vllm_runtime

        prepare_vllm_runtime()
        return VLLMLogprobClient(model_key)
    raise SystemExit(f"unknown --client {kind}")


def main() -> None:
    spec = load_sweep_spec()
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=("constant", "vllm"), required=True)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--dataset", action="append", dest="datasets")
    ap.add_argument("--paradigm", choices=("A", "B"), default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    datasets = args.datasets or list(spec["datasets"])
    conditions = spec["paradigms"]
    if args.paradigm:
        conditions = [c for c in conditions if c["paradigm"] == args.paradigm]
    client = _client(args.client, args.model)

    cfg = {
        "client": args.client,
        "model": args.model,
        "datasets": datasets,
        "conditions": conditions,
        "sweep": spec,
    }
    rows: list[dict] = []
    for dataset in datasets:
        for cond in conditions:
            scheme = cond.get("scheme")
            rows.extend(
                run_dataset_sweep(
                    dataset,
                    client,
                    paradigm=cond["paradigm"],
                    scheme=scheme,
                    model_key=args.model,
                )
            )

    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["model"], r["paradigm"], r["scheme"]), []).append(r)
    winners = [pick_winner(v) for v in grouped.values()]

    dest = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "prompt_sweep"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        **run_metadata(cfg),
        "cells": rows,
        "winners": winners,
        "frozen": False,
        "note": "Winners are candidates only. Do not write a lockfile from --client constant.",
    }
    out = dest / f"sweep_{args.client}_{args.model}.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"n_cells": len(rows), "n_winners": len(winners), "wrote": str(out)}, indent=2))


if __name__ == "__main__":
    main()
