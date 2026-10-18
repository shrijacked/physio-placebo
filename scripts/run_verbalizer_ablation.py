"""Score the three Week 9 sentence wordings. Prompts stay the frozen ones.

    python scripts/run_verbalizer_ablation.py --client vllm --model qwen3_8b
    python scripts/run_verbalizer_ablation.py --client vllm --model llama31_8b --dataset wesad --variant ii

Variant (i) is scored again so each minute can be paired with (iii). SensorLM
is not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.eval_grid import run_eval_cell
from physio_placebo.prompts.freeze import load_frozen_prompts
from physio_placebo.provenance import run_metadata
from physio_placebo.scoring.client import ConstantClient

VARIANTS = ("i", "ii", "iii")


def _client(kind: str, model_key: str):
    if kind == "constant":
        return ConstantClient()
    if kind == "vllm":
        from physio_placebo.scoring.vllm_client import VLLMLogprobClient, prepare_vllm_runtime

        prepare_vllm_runtime()
        return VLLMLogprobClient(model_key)
    raise SystemExit(f"unknown client {kind}")


def _datasets(model: str, only: str | None) -> list[str]:
    found = []
    for winner in load_frozen_prompts()["winners"]:
        if winner["model"] == model and winner["paradigm"] == "B" and winner["dataset"] not in found:
            found.append(winner["dataset"])
    if only is not None:
        if only not in found:
            raise SystemExit(f"{model} has no frozen Paradigm B cell for {only}")
        return [only]
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=("constant", "vllm"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    variants = args.variant or list(VARIANTS)
    root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "verbalizer_ablation"
    client = _client(args.client, args.model)
    for dataset in _datasets(args.model, args.dataset):
        for variant in variants:
            out = root / args.model / dataset / variant / "cell.json"
            if out.exists():
                print(f"skip {out}")
                continue
            cell = run_eval_cell(
                dataset,
                client,
                model_key=args.model,
                paradigm="B",
                scheme=None,
                verbalizer_variant=variant,
                save_rows=True,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **run_metadata(
                    {
                        "client": args.client,
                        "model": args.model,
                        "dataset": dataset,
                        "variant": variant,
                    }
                ),
                "cell": cell,
            }
            out.write_text(json.dumps(payload, default=str) + "\n")
            print(
                f"wrote {out} variant={variant} macro_f1={cell['macro_f1']:.4f} "
                f"n={cell['n_scored']} failures={cell['n_failures']}"
            )


if __name__ == "__main__":
    main()
