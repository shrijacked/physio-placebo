# Health-LLM: released-code and paper notes

**Purpose (Week 1 exit criterion + Week 3 fidelity anchor).** Summarize exactly what the released
Health-LLM pipeline does, verify the paper's headline numbers we anchor against, and document every
gap between the paper's claims and the released code. Everything below was verified directly against
the repository clone at `data/third_party/Health-LLM` (shallow clone, 2026-08-27) and the paper PDF
shipped inside it (`pdf/paper.pdf`).

- Paper: Kim et al., *Health-LLM: Large Language Models for Health Prediction via Wearable Sensor
  Data*, CHIL 2024. Repo: `mitmedialab/Health-LLM`.
- Our anchor task: **PMData stress prediction** (self-reported stress 1–5, MAE, lower is better).

---

## 1. What the pipeline actually is

Health-LLM is **not** a signal-processing pipeline. The wearable "sensor data" is reduced to a
handful of daily summary statistics (steps, calories, resting HR, sleep minutes, self-reported mood),
templated into an English question, and the LLM answers with a number.

### 1.1 Dataset generation (`gen_dataset.py`)

For PMData stress (the anchor task), per wellness-survey row (lines 504–538):

- **Inputs in the prompt** (line 518): the last 14 days of per-event lists — `[Steps]`,
  `[Burned Calorories]` (sic), `[Resting Heart Rate]`, `[SleepMinutes]` — plus **`[Mood]: m out of 5`**.
- **Label** (line 528): `stress` (1–5) from the **same wellness.csv row** as the mood input.
- Instruction: `"You are a personalized healthcare agent trained to predict stress which ranges
  from 1 to 5 based on physiological data and user information."`

Train/eval split (lines 725–784): `random.seed(123)`, **record-level shuffle**, first 50% → train
pool, eval = up to **299 items** taken from the second half. **No subject field survives into the
JSON; no subject-aware splitting of any kind.**

### 1.2 Inference (`inference.py`)

- Prompt modes: zero-shot / few-shot / few-shot CoT / CoT-SC (lines 93–215).
- Few-shot exemplars (line 156): `random.choice(range(len(data))[50:])` — sampled **from the same
  eval JSON that is being evaluated**, so exemplars can be other eval items (or, up to the format
  string, the query itself). Combined with the record-level split, few-shot results have both
  **item-level and subject-level leakage**.
- The "format prompt" (lines 117–144) shows the model a **random valid label** as the example answer
  (`Answer: 3`), a mild answer-priming confound. For stress it samples from `{0..5}` although the
  declared label range is 1–5.
- Scoring: free-text generation, parsed later (separate eval scripts) — no logprob scoring.

---

## 2. Verified headline numbers (paper Table 3, page 6; MAE ↓, PMData STRS)

| Model | Zero-shot | Few-shot |
|---|---|---|
| MedAlpaca-7b | **0.76 ± 0.1** ← anchor | 0.78 ± 0.1 |
| GPT-3.5 (`gpt-3.5-turbo-instruct`) | — (failed) | 0.94 ± 0.1 |
| GPT-4 | — (failed) | 0.76 ± 0.1 |
| Gemini-Pro (1.0) | 0.79 ± 0.0 | 1.10 ± 0.0 |
| GPT-4 few-shot + CoT-SC (their best prompt-engineered) | | 0.33 ± 0.1 |
| HealthAlpaca-13b (their fine-tuned model, Table 4) | 0.31 | |

Appendix Table 17 (page 21) re-reports the same grid in MAPE/macro-F1; Table 16 (page 21) gives
dataset sizes and label distributions.

### 2.1 The majority-class observation (our thesis, in their own numbers)

Table 16 reports the PMData stress label distribution:
`{0: 1, 1: 21, 2: 315, 3: 833, 4: 240, 5: 8}` (sums to 1,418, though the same row states N=1,784 —
an internal inconsistency in the paper).

A **constant predictor that always answers "3"** achieves, on that distribution:

MAE = (1·3 + 21·2 + 315·1 + 833·0 + 240·1 + 8·2) / 1418 = 616 / 1418 = **0.434**

That beats zero-shot MedAlpaca (0.76), zero-shot Gemini-Pro (0.79), few-shot GPT-3.5 (0.94), and
few-shot GPT-4 (0.76). Only the heavily prompt-engineered GPT-4 CoT-SC (0.33) and their fine-tuned
HealthAlpaca (0.31) beat the constant — and not by much. **The paper never reports a majority-class
or constant baseline.** This is precisely the placebo-shaped gap this study is designed to measure.

### 2.2 Label-adjacent input ("mood")

The prompt for stress includes same-survey, same-day self-reported **mood (1–5)**. Mood and stress
are near-collinear self-reports from the same instrument (PMData `wellness.csv`). A model could
ignore every sensor value and read mood alone. Any reproduction gap analysis must keep this in mind:
the task, as released, is substantially "predict one self-report from another".

---

## 3. Released-code defects (verified line-by-line)

`inference.py` **cannot run as released** for any model:

1. `argparse` used (line 16) but never imported → `NameError` at import time.
2. `requests` used (line 70, GPT-4 branch) but never imported.
3. `get_response` dispatch is broken (lines 35–90): the caller passes a
   `genai.GenerativeModel` object (lines 179, 206, 220, 238), but the function compares
   `model == 'gemini-pro'` (string compare → always False for an object), falls through to
   `"medAlpaca" in model` → `TypeError`. Because every call site wraps the call in
   `while True: try/except: continue`, the released script **infinite-loops printing exceptions**
   rather than crashing.
4. The `--model` CLI argument (default `gpt-3.5`) is **ignored**: the main loop hardcodes
   `genai.GenerativeModel('gemini-pro')` (lines 220, 238) and writes to `output/gemini-pro/...`
   (lines 105, 260) regardless.
5. `medalpaca_pl` (line 85) is referenced but never defined — the MedAlpaca path relies on code
   that is not in the repository.
6. `count += 1` inside an `except` block (line 184) with `count` never initialized →
   `NameError` if that branch is ever reached.
7. Requires **both** `openai_key` and `genai_key` env vars unconditionally (lines 22–23), even for
   local models.
8. GPT-3.5 branch uses the pre-1.0 `openai.Completion.create(engine=...)` API (lines 75–76),
   removed from the OpenAI SDK in Nov 2023.

`gen_dataset.py` additionally:

9. PMData path crashes on the real dataset: `participant_info` contains only a stub for `p1`
   (lines 334–336) and `participant_info[dir1]` raises `KeyError` for `p2..p16` (line 345). The
   released script cannot process the full PMData cohort without editing.
10. The paper's stated token length (202 ± 42) is inconsistent with the prompt construction at
    line 518, which dumps raw Python lists of up to 14 days of per-event values into the prompt.

Found while executing the fallback anchor (2026-08-30):

11. **The two released scripts are mutually incompatible.** The generator's eval writer renames
    `input`→`question` and `output`→`answer` and deletes the originals (lines 777–781), but
    `inference.py` consumes `_data['input']` / `_data['output']` (lines 114–115). Feeding the
    generator's eval file to their inference script raises `KeyError`; a schema adapter is
    required (ours restores the keys, preserving order; disclosed in `results/anchor/`).
12. **`set_seed()` is defined (line 25) but never called.** The paper's "seeds {0, 1, 2}"
    therefore cannot have controlled exemplar sampling or anything else; as released, the seed
    only selects the output filename and the three runs are unseeded repetitions.
13. **Missing-file handling bleeds data across participants.** In the official Simula PMData
    release, p12 and p13 have no `fitbit/resting_heart_rate.json`. The generator's bare
    `except: continue` (line 367) skips the remaining Fitbit loads for that participant, after
    which the aggregation runs on whatever `exercise_data`/`sleep_data`/`heart_rate_data`
    variables are still in scope — a `NameError` crash if the unlucky participant comes first,
    or the **previous participant's sensor data** paired with the current participant's labels
    otherwise (dependent on `os.listdir` order, so irreproducible either way).

**Implication for the fidelity anchor:** reproducing "their released code" necessarily means a
minimally-patched version. Every patch will be recorded as a diff in `docs/fidelity-anchor.md`, and
the reproduction gap reported with those patches disclosed.

---

## 4. Methodological gaps relevant to our design

| Gap in Health-LLM | Our design's answer |
|---|---|
| Record-level random split; same subjects in train/eval | Strict LOSO; automated leakage test |
| Few-shot exemplars drawn from the eval pool | Exemplars only from held-out prompt-dev subjects, excluded from evaluation |
| No majority-class or permutation chance floor | Both, locked in Week 2 before any LLM runs |
| No signal-ablation control of any kind | 5-rung surrogate ladder + information-preservation ledger |
| Label-adjacent self-report (mood) in inputs | Sensor-only features, frozen extraction code |
| Free generation + text parsing | Single-forward-pass logprobs over label tokens |
| MAE on ordinal self-report labels | Macro-F1 on protocol-defined binary tasks |

## 5. Anchor decision (Week 3)

- **Primary anchor: PMData stress, zero-shot MedAlpaca-7b, MAE 0.76 ± 0.1** (Table 3).
  Rationale: open weights (`medalpaca/medalpaca-7b` on HF), no dependence on retired APIs
  (their zero-shot GPT-3.5/GPT-4 stress cells are "—" failures; Gemini-Pro 1.0 is retired),
  zero-shot avoids the few-shot exemplar-leakage confound.
- Fallback: few-shot GPT-3.5 `gpt-3.5-turbo-instruct` (0.94 ± 0.1) only if that engine is still
  served and an OpenAI key is provided. **Update 2026-08-30: both conditions now hold** (engine
  present in the account's model list; key in git-ignored `.env`).
- Reproduction target defined as: run their (patched) pipeline on PMData stress eval split
  (seed 123, ≤299 items), MedAlpaca-7b, zero-shot, 3 seeds {0,1,2}, report MAE mean ± sd and the
  gap vs 0.76.
