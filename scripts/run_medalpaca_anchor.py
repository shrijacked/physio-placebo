"""Run the Week-3 Health-LLM fidelity anchor (zero-shot MedAlpaca-7b, PMData stress).

Primary path: their released ``inference.py`` zero-shot loop, with ``medalpaca_pl``
defined as a transformers text-generation pipeline on ``medalpaca/medalpaca-7b``
fp16 / CUDA device 0. HuggingFace returns a list; the wrapper makes the released
call site ``medalpaca_pl(question)['generated_text']`` work. Completions only
(``return_full_text=False``) so first-number scoring does not read sensor values
out of the prompt.

This Mac cannot run the inference (no CUDA, 16 GB unified, ~14 GB free disk).
``--dry-run`` builds the three seeded prompt sets and writes a prompt manifest
without loading weights. ``--out-dir`` writes into an existing folder and resumes
from ``stress_sd{seed}.partial.json`` (needed on Colab).

Usage:
    python scripts/run_medalpaca_anchor.py --dry-run
    python scripts/run_medalpaca_anchor.py          # CUDA required
    python scripts/run_medalpaca_anchor.py --out-dir /content/drive/MyDrive/physio-placebo-anchor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physio_placebo.anchor import (
    DEFAULT_EVAL_JSON,
    PAPER_MEDALPACA,
    SEEDS,
    aggregate_seed_maes,
    build_zero_shot_prompts,
    load_eval_items,
    locked_eval_sha256,
    remaining_prompts,
    score_records,
)
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import config_hash, run_metadata, sha256_bytes, sha256_file, utc_stamp

HF_MODEL = "medalpaca/medalpaca-7b"
MAX_NEW_TOKENS = 120
CFG = {
    "model": HF_MODEL,
    "mode": "zero-shot",
    "task": "pmdata_stress",
    "seeds": list(SEEDS),
    "max_new_tokens": MAX_NEW_TOKENS,
    "torch_dtype": "float16",
    "return_full_text": False,
    "do_sample": False,
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _out_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        dest = explicit if explicit.is_absolute() else REPO_ROOT / explicit
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    dest = REPO_ROOT / "results" / "anchor" / "medalpaca" / f"{utc_stamp()}_{config_hash(CFG)[:8]}"
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def _write_prompts(out_dir: Path, items) -> dict:
    manifest = {
        "eval_file": str(DEFAULT_EVAL_JSON.relative_to(REPO_ROOT)),
        "eval_sha256": locked_eval_sha256(),
        "n_items": len(items),
        "seeds": {},
    }
    for seed in SEEDS:
        prompts = build_zero_shot_prompts(items, seed)
        payload = json.dumps(prompts, indent=2) + "\n"
        path = out_dir / f"prompts_sd{seed}.json"
        path.write_text(payload)
        manifest["seeds"][str(seed)] = {
            "n": len(prompts),
            "sha256": sha256_bytes(payload.encode()),
            "path": str(_rel(path)),
        }
    (out_dir / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_pipeline():
    import torch
    from transformers import pipeline as hf_pipeline

    if not torch.cuda.is_available():
        raise SystemExit(
            "MedAlpaca-7b fp16 needs CUDA (~14 GB). This machine has none. "
            "Re-run with --dry-run, or on a >=16 GB GPU box."
        )

    pipe = hf_pipeline(
        "text-generation",
        model=HF_MODEL,
        torch_dtype=torch.float16,
        device=0,
    )
    pipe.tokenizer.pad_token_id = 0

    def medalpaca_pl(question: str) -> dict:
        # PATCH P2: released call site is medalpaca_pl(question)['generated_text'].
        # HF pipeline returns a list; wrap. return_full_text=False so scoring reads
        # the completion, not the step-count numbers in the prompt.
        out = pipe(
            question,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
            do_sample=False,
        )
        return out[0]

    return medalpaca_pl


def _run_inference(out_dir: Path, items) -> dict:
    medalpaca_pl = _load_pipeline()
    per_seed = {}
    for seed in SEEDS:
        prompts = build_zero_shot_prompts(items, seed)
        pred_path = out_dir / f"stress_sd{seed}.json"
        partial_path = out_dir / f"stress_sd{seed}.partial.json"
        source = pred_path if pred_path.is_file() else partial_path
        records = json.loads(source.read_text()) if source.is_file() else []
        todo = remaining_prompts(prompts, records)
        if records and todo:
            print(f"seed {seed}: resuming at item {len(records) + 1}/{len(prompts)}")
        for prompt in todo:
            try:
                answer = medalpaca_pl(prompt["question"])["generated_text"]
            except Exception as exc:  # noqa: BLE001 — match their N/A swallow, but log it
                print(f"seed={seed} no={prompt['no']}: {exc}", file=sys.stderr)
                answer = "N/A"
            records.append(
                {
                    "no": prompt["no"],
                    "question": prompt["question"],
                    "answer": answer,
                    "label": prompt["label"],
                }
            )
            partial_path.write_text(json.dumps(records, indent=2) + "\n")
        pred_path.write_text(json.dumps(records, indent=2) + "\n")
        if partial_path.is_file():
            partial_path.unlink()
        scored = score_records(records)
        scored["pred_file"] = str(pred_path)
        scored["pred_file_sha256"] = sha256_file(pred_path)
        per_seed[seed] = scored
        print(
            f"seed {seed}: MAE={scored['mae']:.4f}  "
            f"parsed={scored['n_parsed']}/{scored['n_items']}  "
            f"failures={scored['n_parse_failures']}"
        )
    return per_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build seeded zero-shot prompts only; do not load MedAlpaca",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write/resume directory (use a Drive path on Colab so disconnects don't wipe work)",
    )
    args = ap.parse_args()

    items = load_eval_items()
    out_dir = _out_dir(Path(args.out_dir) if args.out_dir else None)
    prompt_manifest = _write_prompts(out_dir, items)
    meta = {
        **run_metadata(CFG),
        "eval_sha256": locked_eval_sha256(),
        "paper_target": PAPER_MEDALPACA,
        "dry_run": args.dry_run,
        "prompt_manifest": prompt_manifest,
    }

    if args.dry_run:
        (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"dry-run: wrote {len(SEEDS)} prompt files under {_rel(out_dir)}")
        print("inference still needs a CUDA GPU; this is not the Week-3 number.")
        return

    per_seed = _run_inference(out_dir, items)
    mae_mean, mae_sd = aggregate_seed_maes(per_seed)
    meta["per_seed"] = {str(k): v for k, v in per_seed.items()}
    meta["mae_mean"] = round(mae_mean, 4)
    meta["mae_sd"] = round(mae_sd, 4)
    meta["gap_vs_paper"] = round(mae_mean - PAPER_MEDALPACA["mae_mean"], 4)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"MAE mean +/- sd: {mae_mean:.3f} +/- {mae_sd:.3f}")
    print(f"paper target: {PAPER_MEDALPACA['mae_mean']} +/- {PAPER_MEDALPACA['mae_sd']}")
    print(f"gap: {mae_mean - PAPER_MEDALPACA['mae_mean']:+.3f}")
    print(f"wrote {_rel(out_dir)}")


if __name__ == "__main__":
    main()
