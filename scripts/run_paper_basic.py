"""Zero-shot MedAlpaca on the paper's basic stress prompt.

Table 1 basic context is the 14-day average, not the raw list. Figure 7 is the
zero-shot frame. Figure 8 greedy decoding: 1 beam, do_sample false, 128 new
tokens. The locked 2.689 run is a different prompt and is not overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physio_placebo.anchor import (
    load_eval_items,
    medalpaca_generate_prompt,
    paper_basic_prompt,
    paper_basic_question,
    paper_figure1_prompt,
    paper_figure7_list_prompt,
    parse_first_number,
    parse_label,
    score_records,
)
from physio_placebo.paths import REPO_ROOT

HF_MODEL = "medalpaca/medalpaca-7b"


def _prompts(item_input: str) -> dict[str, tuple[str, str]]:
    question = paper_basic_question(item_input)
    return {
        "figure7": (paper_basic_prompt(item_input), "Response:"),
        "figure1": (paper_figure1_prompt(item_input), "### Response:"),
        "figure7list": (paper_figure7_list_prompt(item_input), "Response:"),
        "medalpaca": (
            medalpaca_generate_prompt(
                instruction="You are an intelligent healthcare agent.",
                input=question,
            ),
            "### Response:",
        ),
    }


def _generate(model, tok, prompt: str, marker: str) -> str:
    import torch
    from transformers import GenerationConfig

    ids = tok(prompt, return_tensors="pt", truncation=False)["input_ids"].to("cuda")
    # Figure 8 greedy search. pad_token_id 0 is their tokenizer assignment;
    # the checkpoint's -1 makes this transformers build return empty text.
    cfg = GenerationConfig(
        pad_token_id=0,
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        max_new_tokens=128,
    )
    with torch.no_grad():
        output = model.generate(input_ids=ids, generation_config=cfg)
    decoded = tok.decode(output[0], skip_special_tokens=True)
    return decoded.split(marker)[-1].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out-dir", default="results/anchor/paper_basic")
    ap.add_argument(
        "--variant",
        action="append",
        choices=("figure7", "figure7list", "figure1", "medalpaca"),
        default=None,
        help="which paper frame to run; default is all three",
    )
    args = ap.parse_args()
    variants = args.variant or ["figure7", "figure1", "medalpaca"]
    items = load_eval_items()
    if args.limit is not None:
        items = items[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    tok.pad_token_id = 0
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_variant: dict[str, list[dict]] = {name: [] for name in variants}
    for item in items:
        built = _prompts(item["input"])
        label = item["output"]
        for name in variants:
            prompt, marker = built[name]
            answer = _generate(model, tok, prompt, marker)
            pred = parse_first_number(answer)
            print(f"{name} label={parse_label(label)} pred={pred} ans={answer[:160]!r}", flush=True)
            per_variant[name].append(
                {
                    "label": label,
                    "answer": answer,
                    "prompt_sent": prompt,
                }
            )
            (out_dir / f"{name}.partial.json").write_text(
                json.dumps(per_variant[name]) + "\n"
            )
    summary = {}
    for name, records in per_variant.items():
        scored = score_records(records)
        summary[name] = scored
        print(
            f"{name}: MAE={scored['mae']:.4f} "
            f"parsed={scored['n_parsed']}/{scored['n_items']}"
        )
        (out_dir / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("WROTE", out_dir)


if __name__ == "__main__":
    main()
