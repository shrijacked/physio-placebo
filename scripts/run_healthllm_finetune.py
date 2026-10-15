"""LoRA-fine-tune MedAlpaca-7b on the locked PMData stress complement, then score it.

The scorer is the locked first-number MAE. This does not touch results/anchor or
results/eval_grid.

    python scripts/run_healthllm_finetune.py --dry-run
    python scripts/run_healthllm_finetune.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from physio_placebo.anchor import (
    build_zero_shot_prompts,
    load_eval_items,
    medalpaca_train_text,
    score_records,
)
from physio_placebo.comparison import load_finetune_lock
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
        / "medalpaca_ft"
        / f"{utc_stamp()}_{config_hash(lock)[:8]}"
    )
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def _tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    features = []
    eos = tokenizer.eos_token_id
    for item in rows:
        text = medalpaca_train_text(item)
        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        if eos is not None and (not ids or ids[-1] != eos) and len(ids) < max_length:
            ids = [*ids, eos]
        features.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": list(ids),
            }
        )
    return features


def _generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(question, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _train_and_score(train_json: Path, out_dir: Path, lock: dict) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit(
            "MedAlpaca-7b fp16 LoRA needs CUDA. Re-run on the A6000, or pass --dry-run."
        )

    rows = json.loads(train_json.read_text())
    tokenizer = AutoTokenizer.from_pretrained(lock["model"], use_fast=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    features = _tokenize_rows(rows, tokenizer, int(lock["model_max_length"]))

    class _Rows(Dataset):
        def __len__(self) -> int:
            return len(features)

        def __getitem__(self, index: int) -> dict:
            return features[index]

    adapter_dir = out_dir / "adapter"
    if (adapter_dir / "adapter_config.json").is_file():
        print(f"adapter exists, skipping training: {adapter_dir}")
    else:
        base = AutoModelForCausalLM.from_pretrained(
            lock["model"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        base.config.use_cache = False
        lora = LoraConfig(
            r=int(lock["lora_r"]),
            lora_alpha=int(lock["lora_alpha"]),
            lora_dropout=float(lock["lora_dropout"]),
            target_modules=list(lock["lora_target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
        if lock.get("gradient_checkpointing"):
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(out_dir / "trainer"),
                per_device_train_batch_size=int(lock["per_device_batch_size"]),
                gradient_accumulation_steps=int(lock["gradient_accumulation_steps"]),
                num_train_epochs=int(lock["num_epochs"]),
                learning_rate=float(lock["learning_rate"]),
                warmup_steps=int(lock["warmup_steps"]),
                lr_scheduler_type=str(lock["lr_scheduler_type"]),
                optim=str(lock["optim"]),
                fp16=True,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                seed=int(lock["seed"]),
                gradient_checkpointing=bool(lock.get("gradient_checkpointing")),
                remove_unused_columns=False,
                dataloader_num_workers=0,
            ),
            train_dataset=_Rows(),
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, padding=True
            ),
        )
        trainer.train()
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        del trainer, model, base
        torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        lock["model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tuned = PeftModel.from_pretrained(base, adapter_dir)
    tuned.eval()
    seed = int(lock["eval_seed"])
    prompts = build_zero_shot_prompts(load_eval_items(), seed)
    pred_path = out_dir / f"stress_sd{seed}.json"
    partial_path = out_dir / f"stress_sd{seed}.partial.json"
    records = json.loads(partial_path.read_text()) if partial_path.is_file() else []
    done = {int(rec["no"]) for rec in records}
    for prompt in prompts:
        if int(prompt["no"]) in done:
            continue
        try:
            answer = _generate(tuned, tokenizer, prompt["question"], int(lock["max_new_tokens"]))
        except Exception as exc:  # noqa: BLE001 — record the failure, keep the item
            print(f"no={prompt['no']}: {exc}", file=sys.stderr)
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
    scored["eval_seed"] = seed
    scored["pred_file"] = str(pred_path.relative_to(REPO_ROOT))
    scored["pred_file_sha256"] = sha256_file(pred_path)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    lock = load_finetune_lock()
    train_json = args.train_json or latest_train_json()
    rows = json.loads(train_json.read_text())
    print(
        f"model={lock['model']} epochs={lock['num_epochs']} "
        f"n_train={len(rows)} train_json={train_json}"
    )
    if args.dry_run:
        print("dry-run: lock loaded, train rows readable, no weights loaded")
        return
    out_dir = _out_dir(lock, args.out_dir)
    scored = _train_and_score(train_json, out_dir, lock)
    manifest = {
        "train_json": str(train_json),
        "train_sha256": sha256_file(train_json),
        "n_train": len(rows),
        "mae": scored["mae"],
        "score": scored,
        **run_metadata(lock),
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MAE {scored['mae']:.6f}  parsed {scored['n_parsed']}/{scored['n_items']}  dir={out_dir}")


if __name__ == "__main__":
    main()
