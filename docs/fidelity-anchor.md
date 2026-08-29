# Fidelity anchor — reproducing one Health-LLM headline number

**Status: NOT RUN.** This document currently records the anchor choice, the exact reproduction
recipe, and the patches the released code needs. The reproduction number and gap will be added when
the run executes (Week 3). Nothing below has been executed against PMData yet.

## 1. Anchor target

> **PMData stress prediction, zero-shot, MedAlpaca-7b: MAE 0.76 ± 0.1**
> (Health-LLM, CHIL 2024, Table 3, page 6; mean ± sd over seeds {0, 1, 2}.)

Why this cell (full reasoning in `docs/health-llm-notes.md` §5):

- Open weights — `medalpaca/medalpaca-7b` on Hugging Face; no retired-API risk. The paper's own
  zero-shot GPT-3.5/GPT-4 stress cells are failures ("—"), and Gemini-Pro 1.0 no longer exists,
  so the API cells are unreproducible in 2026 by anyone.
- Zero-shot avoids the few-shot exemplar-leakage confound in their released `inference.py`.

Fallback — **VIABLE as of 2026-08-30** (user's OpenAI key stored in git-ignored `.env`;
`gpt-3.5-turbo-instruct` verified present in the account's model list): few-shot GPT-3.5,
MAE 0.94 ± 0.1. Runs API-only on this Mac (≈$1). Caveat: this cell inherits their few-shot
exemplar-leakage confound, which is disclosed wherever the number is reported.

## 2. Reproduction recipe (planned, not yet run)

1. **Data:** download PMData (16 participants, Fitbit Versa 2 + PMSys self-reports) from
   Simula: <https://datasets.simula.no/pmdata/>. Only `pX/pmsys/wellness.csv` and
   `pX/fitbit/{exercise,resting_heart_rate,sleep}.json` are consumed by their generator.
2. **Generate eval split with their code:** `gen_dataset.py`, `DATA="PMData"`,
   `SUBTASK="stress"`, seed 123 record-shuffle, eval = ≤299 items from the second half
   (their lines 716–784), written to `eval/data/pmdata_stress/step1.json`.
3. **Inference with their (patched) code:** zero-shot mode of `inference.py`, MedAlpaca-7b via
   `transformers` pipeline, seeds {0, 1, 2}, `max_tokens≈120`.
4. **Scoring:** the repository releases **no evaluation code at all** (verified 2026-08-27:
   no `eval/` directory; `medalpaca/` holds only training/inference utilities). The MAE the
   paper reports must be re-implemented: we will parse the first number in the generated
   answer, drop unparsable items, and report MAE with the parse-failure rate. This
   re-implementation is itself part of the reproduction gap and is disclosed as such.
5. **Report:** MAE mean ± sd over 3 seeds, plus reproduction gap vs 0.76, plus (our addition,
   clearly separated) the constant-3 baseline MAE ≈ 0.434 on their own label distribution.

## 3. Required patches to their released code (to be recorded as diffs when run)

The released scripts cannot run as-is; every line reference is verified in
`docs/health-llm-notes.md` §3. Minimum patch set:

| # | File | Patch | Why |
|---|---|---|---|
| P1 | `inference.py` | add `import argparse` | NameError at startup |
| P2 | `inference.py` | define `medalpaca_pl = pipeline("text-generation", model="medalpaca/medalpaca-7b", ...)` | referenced but never defined |
| P3 | `inference.py` | route the main loop through `args.model` instead of hardcoded `genai.GenerativeModel('gemini-pro')`; fix output dir | `--model` is ignored as released |
| P4 | `inference.py` | make the API-key reads conditional on the chosen model | demands OpenAI+Google keys even for local models |
| P5 | `gen_dataset.py` | fill `participant_info` for p1–p16 (age/height/gender placeholders identical to their `p1` stub) | KeyError for every participant except p1 |

Patch policy: **only** changes needed to make their pipeline execute; no methodological
improvements. The reproduction gap is reported with these patches disclosed.

## 4. Compute plan

MedAlpaca-7b needs ~14 GB in fp16. **This Mac is ruled out** (verified 2026-08-27: 16 GB
unified memory, 13 GiB free disk — neither holds the fp16 weights). Options:

1. A GPU box (16 GB card is sufficient) — required for the primary (MedAlpaca) run. Note: no
   longer implied by the main grid, which moved to the OpenAI API on 2026-08-30; a GPU must be
   provisioned specifically for this run (or the run waits).
2. Last resort only: 4-bit quantized weights locally (~4 GB) — this deviates from their
   `transformers` fp16 pipeline and would confound the reproduction; if ever used, the
   quantization is disclosed next to the number and the run is repeated on the GPU box.

## 5. User actions required

- [x] PMData: public zip is only 1.4 GB — agent download started 2026-08-27 into
      `data/raw/pmdata/pmdata.zip` (sha256 recorded on completion).
- [ ] Provide the GPU box (§4) — needed for the **primary** (MedAlpaca) run only.
- [x] OpenAI key provided 2026-08-30 (git-ignored `.env`) — the fallback anchor is runnable
      now, on this Mac, without a GPU.

## 6. Results (to be filled by the run)

| Quantity | Paper | Our reproduction | Gap |
|---|---|---|---|
| PMData stress, zero-shot MedAlpaca-7b, MAE | 0.76 ± 0.1 | — | — |
| Parse-failure rate | not reported | — | — |
| Constant-3 baseline MAE (our addition) | not reported | 0.434 (from their Table 16 distribution) | — |
