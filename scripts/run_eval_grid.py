"""Real-signal A/B eval grid. Requires the frozen prompt lockfile.

    python scripts/run_eval_grid.py --client constant --model qwen3_8b --paradigm B
    python scripts/run_eval_grid.py --client vllm --model qwen3_8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.eval_grid import run_eval_cell, run_model_eval
from physio_placebo.prompts.freeze import load_frozen_prompts
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
    lock = load_frozen_prompts()
    if not lock.get("frozen"):
        raise SystemExit("configs/frozen_prompts.lock.yaml is not frozen")
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=("constant", "vllm"), required=True)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--paradigm", choices=("A", "B"), default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    client = _client(args.client, args.model)
    if args.dataset or args.paradigm:
        rows = []
        for w in lock["winners"]:
            if w["model"] != args.model:
                continue
            if args.dataset and w["dataset"] != args.dataset:
                continue
            if args.paradigm and w["paradigm"] != args.paradigm:
                continue
            rows.append(
                run_eval_cell(
                    w["dataset"],
                    client,
                    model_key=args.model,
                    paradigm=w["paradigm"],
                    scheme=w["scheme"],
                )
            )
    else:
        rows = run_model_eval(client, args.model)

    dest = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "eval_grid"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        **run_metadata({"client": args.client, "model": args.model, "n_cells": len(rows)}),
        "frozen": True,
        "lock": "configs/frozen_prompts.lock.yaml",
        "cells": rows,
        "note": "Eval subjects only. Prompt-dev subjects are excluded. Do not start Week 6.",
    }
    out = dest / f"eval_{args.client}_{args.model}.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"n_cells": len(rows), "wrote": str(out)}, indent=2))


if __name__ == "__main__":
    main()
