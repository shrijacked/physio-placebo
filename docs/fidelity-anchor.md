# Fidelity anchor — reproducing one Health-LLM headline number

**Status: fallback run ATTEMPTED 2026-08-30 — blocked by account rate limits at 35/897 calls.**
Everything before the API is done and locked: PMData extracted, their generator patched and run
(299-item eval split, seed 123, `results/anchor/split_manifest.json`), their inference script
patched (`results/anchor/patches.diff`) and made checkpoint/resumable, scorer written
(`scripts/score_anchor.py`). The OpenAI org is free-tier: **50 requests/day/model** — the run
needs 897. Resumes with one command after a billing upgrade (~25 min, ≈$1.30). The primary
(MedAlpaca) run remains GPU-gated. §6 is filled when a run completes.

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

## 2. Reproduction recipe (fallback executed through step 2 on 2026-08-30; step 3 rate-limited)

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

## 3. Patches applied to their released code (as executed for the fallback run, 2026-08-30)

The released scripts cannot run as-is; every line reference is verified in
`docs/health-llm-notes.md` §3. Full unified diff: `results/anchor/patches.diff`
(patched copies live next to the originals as `*_patched.py`; originals untouched).

| # | File | Patch | Why |
|---|---|---|---|
| G1–G3 | `gen_dataset.py` | select `DATA="PMData"`, `SUBTASK="stress"`; fill `participant_info` for p01–p16 with placeholders identical to their `p1` stub (none of these fields reach the prompt) | hardcoded to LifeSnaps sleep_quality; KeyError for every real participant dir |
| G4 | `gen_dataset.py` | initialize per-participant channel variables and skip participants with missing files | p12/p13 lack `resting_heart_rate.json`; as released this crashes or silently pairs the previous participant's sensors with the current labels (notes §3.13) |
| adapter | — | map eval keys `question`/`answer` back to `input`/`output` | their generator and inference scripts are mutually incompatible (notes §3.11) |
| I1–I2 | `inference.py` | add missing `import argparse`; drop unconditional `google.generativeai`/`torch`/`transformers` imports | NameError at startup; unrelated heavyweight deps demanded for an API-only run |
| I3–I4 | `inference.py` | OpenAI SDK ≥1.0 client; same call, same params (`gpt-3.5-turbo-instruct`, `max_tokens=120`, API-default temperature) | pre-1.0 `openai.Completion.create(engine=...)` removed from the SDK Nov 2023 |
| I5 | `inference.py` | scope loops to the anchor cell (few-shot, PMData stress) | avoid paying for the other 47 mode×task cells |
| I6 | `inference.py` | call their `set_seed(seed)` at the top of each seed iteration | defined but never called as released; the paper's "seeds {0,1,2}" otherwise control nothing (notes §3.12) |
| I7–I8 | `inference.py` | route through `args.model` with bounded (5×) retries; write outputs under `output/gpt-3.5/` and create the directory | `--model` ignored as released (everything hardcodes gemini-pro); infinite retry loop cannot terminate on persistent errors |

Patch policy: **only** changes needed to make their pipeline execute; no methodological
improvements. The reproduction gap is reported with these patches disclosed. P2 from the
original plan (define `medalpaca_pl`) is still pending — it belongs to the GPU-gated
primary run only.

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
- [ ] **Upgrade the OpenAI account past the free tier** (add payment method / ≥$5 credits at
  platform.openai.com → Settings → Billing): the org currently allows 50 requests/day/model
  (verified 2026-08-30 — killed the fallback run at call 50 and equally caps the gpt-4.1
  models the main grid uses).

## 6. Results (to be filled by the run)

| Quantity | Paper | Our reproduction | Gap |
|---|---|---|---|
| PMData stress, zero-shot MedAlpaca-7b, MAE (primary) | 0.76 ± 0.1 | — (GPU-gated) | — |
| PMData stress, few-shot `gpt-3.5-turbo-instruct`, MAE (fallback) | 0.94 ± 0.1 | — (rate-limited at 35/897 calls) | — |
| Parse-failure rate | not reported | — | — |
| Constant-3 baseline MAE (our addition) | not reported | 0.434 on their Table 16 distribution; **0.401 on our locked 299-item eval split** | — |
