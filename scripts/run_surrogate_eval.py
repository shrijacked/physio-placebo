"""Score frozen prompts on one surrogate. Raw logprobs are kept on each row.

    python scripts/run_surrogate_eval.py --client vllm --model qwen3_8b --kind gaussian
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from physio_placebo.data.windows import load_subject_windows
from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.eval_grid import run_eval_cell
from physio_placebo.prompts.freeze import load_frozen_prompts, load_frozen_prompts_c
from physio_placebo.provenance import run_metadata
from physio_placebo.scoring.client import ConstantClient
from physio_placebo.surrogates.ladder import ALL_SURROGATES


def _client(kind: str, model_key: str):
    if kind == "constant":
        return ConstantClient()
    if kind == "vllm":
        from physio_placebo.scoring.vllm_client import VLLMLogprobClient, prepare_vllm_runtime

        prepare_vllm_runtime()
        return VLLMLogprobClient(model_key)
    raise SystemExit(f"unknown client {kind}")


def _windows(dataset: str, signal: str) -> dict | None:
    if signal == "deleted":
        return None
    root = REPO_ROOT / "results" / "surrogates" / "windows" / dataset / signal
    found = {path.stem: load_subject_windows(path) for path in sorted(root.glob("*.npz"))}
    if not found:
        raise FileNotFoundError(root)
    return found


def _features(dataset: str, signal: str):
    if signal == "deleted":
        return None
    path = REPO_ROOT / "results" / "surrogates" / "features" / dataset / signal / "features.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", choices=("constant", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=ALL_SURROGATES, action="append", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--paradigm", choices=("A", "B", "C"), default=None)
    args = parser.parse_args()
    use_c = args.model == "qwen25_vl" or args.paradigm == "C"
    lock = load_frozen_prompts_c() if use_c else load_frozen_prompts()
    winners = []
    for winner in lock["winners"]:
        if winner["model"] != args.model:
            continue
        if args.dataset and winner["dataset"] != args.dataset:
            continue
        if args.paradigm and winner["paradigm"] != args.paradigm:
            continue
        winners.append(winner)
    if not winners:
        raise SystemExit("no frozen cells match")
    client = _client(args.client, args.model)
    for signal in args.kind:
        cache_windows: dict[str, dict | None] = {}
        cache_features: dict[str, object] = {}
        for winner in winners:
            dataset = winner["dataset"]
            key = f"{dataset}:{signal}"
            if key not in cache_windows:
                cache_windows[key] = _windows(dataset, signal)
                cache_features[key] = _features(dataset, signal)
            done = (
                REPO_ROOT
                / "results"
                / "surrogates"
                / "eval"
                / args.model
                / dataset
                / winner["paradigm"]
                / str(winner["scheme"] or "none")
                / signal
                / "cell.json"
            )
            if done.is_file():
                print(f"skip {done}")
                continue
            cell = run_eval_cell(
                dataset,
                client,
                model_key=args.model,
                paradigm=winner["paradigm"],
                scheme=winner["scheme"],
                signal=signal,
                feature_frame=cache_features[key],
                windows=cache_windows[key],
            )
            done.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **run_metadata(
                    {
                        "client": args.client,
                        "model": args.model,
                        "dataset": dataset,
                        "paradigm": winner["paradigm"],
                        "scheme": winner["scheme"],
                        "signal": signal,
                    }
                ),
                "cell": cell,
            }
            done.write_text(json.dumps(payload, indent=2, default=str) + "\n")
            print(
                f"{dataset} {winner['paradigm']} {winner['scheme']} {signal}: "
                f"macro_f1={cell.get('macro_f1')} n={cell.get('n_scored')}"
            )


if __name__ == "__main__":
    main()
