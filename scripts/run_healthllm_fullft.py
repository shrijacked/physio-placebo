"""Full fine-tune MedAlpaca-7b on the locked PMData stress complement, then score it.

Every weight is trained, as Health-LLM Appendix A describes. The recipe is
configs/healthllm_fullft.lock.yaml. The LoRA cell and its 2.697 run are untouched.

    python scripts/run_healthllm_fullft.py --dry-run
    python scripts/run_healthllm_fullft.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from physio_placebo.anchor import (
    load_eval_items,
    medalpaca_eval_prompt,
    medalpaca_train_text,
    score_records,
)
from physio_placebo.comparison import load_fullft_lock
from physio_placebo.paths import REPO_ROOT
from physio_placebo.provenance import config_hash, run_metadata, sha256_file, utc_stamp


def latest_train_json() -> Path:
    root = REPO_ROOT / "results" / "comparison" / "pmdata_stress_train"
    if not root.is_dir():
        raise SystemExit(f"missing train split directory: {root}")
    dirs = sorted(path for path in root.iterdir() if (path / "train.json").is_file())
    if not dirs:
        raise SystemExit(f"no train.json under {root}")
    return dirs[-1] / "train.json"


def _out_dir(lock: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        dest = explicit if explicit.is_absolute() else REPO_ROOT / explicit
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    dest = (
        REPO_ROOT
        / "results"
        / "comparison"
        / "medalpaca_fullft"
        / f"{utc_stamp()}_{config_hash(lock)[:8]}"
    )
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def _tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> list[list[int]]:
    # Same as the LoRA cell: MedAlpaca DataHandler.tokenize, no special tokens,
    # EOS appended, loss on every token (train_on_inputs true).
    eos = tokenizer.eos_token_id
    out = []
    for item in rows:
        ids = tokenizer(
            medalpaca_train_text(item),
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        if eos is not None and (not ids or ids[-1] != eos) and len(ids) < max_length:
            ids = [*ids, eos]
        out.append(ids)
    return out


def _train(model, features: list[list[int]], lock: dict, out_dir: Path) -> list[dict]:
    import torch
    from torchao.optim import AdamW8bit
    from transformers import get_cosine_schedule_with_warmup

    accum = int(lock["gradient_accumulation_steps"])
    epochs = int(lock["num_epochs"])
    steps_per_epoch = math.ceil(len(features) / accum)
    total_steps = steps_per_epoch * epochs
    optimizer = AdamW8bit(
        model.parameters(),
        lr=float(lock["learning_rate"]),
        weight_decay=float(lock["weight_decay"]),
        bf16_stochastic_round=bool(lock["bf16_stochastic_round"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(lock["warmup_steps"]),
        num_training_steps=total_steps,
    )
    rng = random.Random(int(lock["seed"]))
    device = next(model.parameters()).device
    log: list[dict] = []
    step = 0
    model.train()
    for epoch in range(epochs):
        order = list(range(len(features)))
        rng.shuffle(order)
        for start in range(0, len(order), accum):
            chunk = order[start : start + accum]
            total_loss = 0.0
            for idx in chunk:
                ids = torch.tensor([features[idx]], device=device)
                loss = model(input_ids=ids, labels=ids).loss
                (loss / len(chunk)).backward()
                total_loss += float(loss.detach())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "loss": total_loss / len(chunk),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            log.append(row)
            print(json.dumps(row), flush=True)
            (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    del optimizer
    torch.cuda.empty_cache()
    return log


def _generate(model, tokenizer, prompt: str, lock: dict, *, sample: bool) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs = {
        "max_new_tokens": int(lock["max_new_tokens"]),
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if sample:
        kwargs.update(
            {
                "temperature": float(lock["sample_temperature"]),
                "top_k": int(lock["sample_top_k"]),
                "top_p": float(lock["sample_top_p"]),
            }
        )
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def _score(model, tokenizer, lock: dict, out_dir: Path, *, sample: bool, seed: int) -> dict:
    import torch

    if sample:
        torch.manual_seed(seed)
    tag = f"sample_sd{seed}" if sample else "greedy"
    records = []
    for i, item in enumerate(load_eval_items()):
        try:
            answer = _generate(model, tokenizer, medalpaca_eval_prompt(item), lock, sample=sample)
        except Exception as exc:  # noqa: BLE001 — record the failure, keep the item
            print(f"{tag} no={i + 1}: {exc}", file=sys.stderr)
            answer = "N/A"
        records.append({"no": i + 1, "answer": answer, "label": item["output"]})
        if (i + 1) % 50 == 0:
            print(f"{tag}: {i + 1}/299", flush=True)
    pred_path = out_dir / f"stress_{tag}.json"
    pred_path.write_text(json.dumps(records, indent=2) + "\n")
    scored = score_records(records)
    scored["pred_file"] = str(pred_path.relative_to(REPO_ROOT))
    scored["pred_file_sha256"] = sha256_file(pred_path)
    print(f"{tag}: MAE={scored['mae']:.4f} parsed={scored['n_parsed']}/{scored['n_items']}", flush=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    lock = load_fullft_lock()
    train_json = args.train_json or latest_train_json()
    rows = json.loads(train_json.read_text())
    print(f"model={lock['model']} full fine-tune epochs={lock['num_epochs']} n_train={len(rows)}")
    if args.dry_run:
        print("dry-run: lock loaded, train rows readable, no weights loaded")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("full fine-tune needs CUDA")
    torch.manual_seed(int(lock["seed"]))
    out_dir = _out_dir(lock, args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(lock["model"], use_fast=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
    features = _tokenize_rows(rows, tokenizer, int(lock["model_max_length"]))
    model = AutoModelForCausalLM.from_pretrained(
        lock["model"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if lock.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    train_log = _train(model, features, lock, out_dir)

    model.eval()
    model.config.use_cache = True
    scores = {"greedy": _score(model, tokenizer, lock, out_dir, sample=False, seed=0)}
    for seed in lock["eval_seeds"]:
        scores[f"sample_sd{seed}"] = _score(model, tokenizer, lock, out_dir, sample=True, seed=int(seed))
    sampled = [scores[f"sample_sd{seed}"]["mae"] for seed in lock["eval_seeds"]]
    manifest = {
        "train_json": str(train_json),
        "train_sha256": sha256_file(train_json),
        "n_train": len(rows),
        "n_steps": len(train_log),
        "first_loss": train_log[0]["loss"],
        "last_loss": train_log[-1]["loss"],
        "scores": scores,
        "sample_mae_mean": sum(sampled) / len(sampled),
        **run_metadata(lock),
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"done greedy={scores['greedy']['mae']:.4f} "
        f"sample_mean={manifest['sample_mae_mean']:.4f} dir={out_dir}"
    )


if __name__ == "__main__":
    main()
