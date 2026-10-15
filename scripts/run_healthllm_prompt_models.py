"""Frozen Qwen3-8B and Llama-3.1-8B on the locked Health-LLM PMData stress prompts.

Same first-number MAE scorer as the MedAlpaca anchor. Different backbone, so
these rows do not replace the MedAlpaca 2.689 number.

    python scripts/run_healthllm_prompt_models.py --dry-run
    python scripts/run_healthllm_prompt_models.py --model qwen3_8b
    python scripts/run_healthllm_prompt_models.py --model llama31_8b
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from physio_placebo.anchor import (
    SEEDS,
    build_zero_shot_prompts,
    load_eval_items,
    score_records,
)
from physio_placebo.paths import REPO_ROOT, configs_dir
from physio_placebo.provenance import config_hash, run_metadata, sha256_file, utc_stamp

MAX_NEW_TOKENS = 120
MODELS = ("qwen3_8b", "llama31_8b")


def _spec(model_key: str) -> dict:
    doc = yaml.safe_load((configs_dir() / "models.yaml").read_text())
    if model_key not in doc["models"]:
        raise SystemExit(f"unknown model {model_key}")
    return doc["models"][model_key]


def _render(tokenizer, question: str, *, qwen: bool) -> str:
    messages = [{"role": "user", "content": question}]
    extra = {"enable_thinking": False} if qwen else {}
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate(model, tokenizer, rendered: str) -> str:
    import torch

    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _run(model_key: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("Qwen3-8B and Llama-3.1-8B fp16 need CUDA. Re-run on the A6000.")
    spec = _spec(model_key)
    cfg = {
        "model_key": model_key,
        "hf_id": spec["hf_id"],
        "seeds": list(SEEDS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "chat_template": True,
        "enable_thinking": False if model_key == "qwen3_8b" else None,
    }
    out_dir = (
        REPO_ROOT
        / "results"
        / "comparison"
        / "prompt_our_models"
        / model_key
        / f"{utc_stamp()}_{config_hash(cfg)[:8]}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    items = load_eval_items()
    per_seed = {}
    for seed in SEEDS:
        prompts = build_zero_shot_prompts(items, seed)
        pred_path = out_dir / f"stress_sd{seed}.json"
        partial_path = out_dir / f"stress_sd{seed}.partial.json"
        records = json.loads(partial_path.read_text()) if partial_path.is_file() else []
        done = {int(rec["no"]) for rec in records}
        for prompt in prompts:
            if int(prompt["no"]) in done:
                continue
            rendered = _render(tokenizer, prompt["question"], qwen=model_key == "qwen3_8b")
            try:
                answer = _generate(model, tokenizer, rendered)
            except Exception as exc:  # noqa: BLE001
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
        records.sort(key=lambda rec: int(rec["no"]))
        pred_path.write_text(json.dumps(records, indent=2) + "\n")
        if partial_path.is_file():
            partial_path.unlink()
        scored = score_records(records)
        scored["pred_file_sha256"] = sha256_file(pred_path)
        per_seed[str(seed)] = scored
        print(f"{model_key} seed {seed} MAE {scored['mae']:.6f}")
    maes = [per_seed[str(seed)]["mae"] for seed in SEEDS]
    mean = sum(maes) / len(maes)
    var = sum((value - mean) ** 2 for value in maes) / (len(maes) - 1)
    summary = {
        "mae_mean": mean,
        "mae_sd": var**0.5,
        "per_seed": per_seed,
        **run_metadata(cfg),
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{model_key} MAE {mean:.6f} ± {summary['mae_sd']:.6f}  dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=MODELS, default=None)
    args = parser.parse_args()
    keys = (args.model,) if args.model else MODELS
    items = load_eval_items()
    for key in keys:
        spec = _spec(key)
        n = len(build_zero_shot_prompts(items, 0))
        print(f"{key} hf_id={spec['hf_id']} prompts_seed0={n}")
    if args.dry_run:
        print("dry-run: prompts build, no weights loaded")
        return
    for key in keys:
        _run(key)


if __name__ == "__main__":
    main()
