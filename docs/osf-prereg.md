# OSF Pre-registration (draft for user review and filing)

> **Instructions:** This text follows the OSF Preregistration template
> (https://osf.io/prereg). The user reviews, files it on OSF, and records the OSF URL in
> `PROGRESS.md`. After filing, §2 of `CURSOR_SEMESTER_AGENT.md` and everything below is frozen;
> deviations must be reported as such in the paper.
>
> **DRAFT STATUS — not yet filed.** Placeholders that must be resolved before filing are marked
> `⟨...⟩`.

---

## Title

Do large language models read the signal or the story? A surrogate-ablation audit of LLM
"physiological reasoning" on wearable sensor data.

## Authors

⟨Shrijak Kumar⟩; supervised by ⟨supervisor name⟩. AI-assisted engineering (Cursor agent) is used
for implementation; all scientific decisions are pre-registered here.

## Description

Recent work (e.g., Health-LLM, CHIL 2024) reports that LLMs can predict health/affective states
from wearable sensor data presented in prompts. These evaluations typically lack subject-aware
splits, chance floors, and — critically — any control that degrades the physiological signal while
preserving the surface form of the prompt. We test whether LLM performance on standard
physiological classification tasks depends on the physiological signal itself, or on non-signal
cues (prompt scaffolding, label priors, verbalizer adjectives). We replace real signals with a
ladder of surrogates (noise, phase-randomized, subject-permuted, flatline, deleted) and measure how
much performance survives. We additionally ablate the verbalizer: neutralized wording and
adversarially mis-signed adjectives separate "reading the number" from "reading the word".

## Hypotheses / directional predictions (pre-registered, to be falsified)

- **P1.** For the raw-numeric-series paradigm (A) and the plot-image paradigm (C), the Surrogate
  Dependence Score (defined below) is **SDS < 0.3** — i.e., less than 30% of the above-chance
  margin is attributable to the physiological signal.
- **P2.** The feature-verbalized paradigm (B) loses **more than 50% of its above-chance margin**
  when template adjectives are neutralized (variant ii vs i).
- **P3.** Adversarially mis-signed adjectives (variant iii) **flip the model's prediction on more
  than 40% of items** relative to variant (i).

A failure of these predictions (performance survives surrogates) is a reportable, publishable
outcome; the analysis plan is symmetric with respect to outcome direction.

---

## Study type

Secondary analysis of existing public datasets; computational experiment. No new human-subjects
data collection.

## Blinding

Not applicable (no participants). Guard against analyst degrees of freedom: all conditions, metrics,
exclusion rules, and the analysis pipeline are fixed here before any LLM inference is run.

## Existing data (transparency declaration)

At the time of filing:

- All datasets are public, previously published (WESAD, MAUS, CogWear).
- Data preparation (windowing), feature extraction (frozen, source hash `702d5f37…`,
  `configs/frozen_features.lock.json`), and the **classical baselines/chance floors have
  already been computed for WESAD and CogWear** (Week-2 artifact, `results/classical_floor/`,
  locked before this filing by design); the MAUS floor will be locked the same way before any
  LLM run on MAUS. Observed floors: WESAD logreg macro-F1 0.918 vs permutation chance ≈ 0.42;
  CogWear logreg 0.569 vs permutation p95 0.541 (weak classical signal — the SDS validity
  gate is expected to bind for many CogWear cells).
- **No LLM inference of any kind has been run on any dataset** at filing time. The LLM
  experimental grid — the subject of this pre-registration — is untouched.
- One reproduction of a published Health-LLM number (fidelity anchor, PMData/MedAlpaca) ⟨has /
  has not⟩ been run at filing time; it uses the original authors' data/task and shares no data or
  code path with our grid.

## Datasets and tasks (fixed)

| Dataset | Source | Subjects | Signals used | Binary task |
|---|---|---|---|---|
| WESAD | UCI | 15 | chest ECG, EDA, Resp @700 Hz; wrist BVP @64 Hz, EDA @4 Hz | stress (TSST) vs non-stress (baseline + amusement) |
| MAUS | IEEE DataPort | 22 | wrist PixArt PPG @100/128 Hz | n-back workload: 0-back (low) vs 2-/3-back (high) |
| CogWear | PhysioNet | 11 pilot (10 usable*) | Empatica E4 BVP @64 Hz, EDA @4 Hz | Stroop (cognitive load) vs resting baseline |

\* CogWear pilot subject 3 has no `cognitive_load/empatica_eda.csv` upstream (verified against
the published SHA256SUMS.txt, 2026-08-27) and is excluded by the missing-channels
data-integrity rule below, before any model run.

Windowing (fixed, `configs/windowing.yaml`): 60 s windows, 30 s stride, cut strictly inside one
labeled protocol segment (no mixed-condition windows); windows containing NaNs dropped and counted.

## Features (fixed, frozen)

NeuroKit2 v0.2.13 extraction, code frozen Week 1 (source hash recorded in
`configs/frozen_features.lock.json`): 8 HRV time/frequency features, 7 EDA features (cvxEDA
tonic/phasic), 3 respiration features (dataset-dependent availability). Classical baselines and the
LLM verbalizer consume **byte-identical** feature matrices (enforced by test).

## Models (fixed)

- Qwen3-8B-Instruct and Llama-3.1-8B-Instruct, 4-bit, served with vLLM.
- Qwen2.5-VL-7B-Instruct for the plot paradigm (WESAD and MAUS only).
- Optional stretch: one GPT-class API model, budget < $50.
- **Scoring: log-probabilities over the label tokens in a single forward pass.** No free
  generation, no answer parsing.

## Paradigms (fixed)

- **A** — raw numeric series in the prompt (downsampled per config).
- **B** — verbalized frozen features (Health-LLM style).
- **C** — matplotlib line-plot image of the window (WESAD, MAUS only).

## Prompt protocol (fixed)

3 templates × {0-shot, 4-shot} per paradigm. Selection on a held-out **prompt-development subject
fold** (subjects excluded from all evaluation folds); the winning prompt per condition is **frozen
before any surrogate run**. Few-shot exemplars come only from prompt-dev subjects. Prompts are
never tuned on surrogate results.

## Manipulated variables — the surrogate ladder (fixed)

Applied identically to every paradigm, replacing the window's signal content:

1. **Gaussian noise**, moment-matched to the **globally pooled** mean/variance (never per-class).
2. **IAAFT** phase-randomized surrogate (per window).
3. **Subject-permuted, label alignment broken**: signal from a different subject AND an
   independently sampled condition. (The label-**preserving** subject permutation is kept as a
   separate descriptive condition — "population vs individual information" — and is **never used
   as a null**.)
4. **Flatline** (constant at pooled mean).
5. **Deleted** — prompt scaffolding only, no signal content.

**Information-preservation ledger:** the frozen L2 logistic regression is run on every surrogate
condition and its macro-F1 published, so each surrogate's actual information retention is measured,
not asserted.

## Verbalizer-leakage ablation (fixed; paradigm B only)

- (i) original adjective templates;
- (ii) neutralized — raw values + population percentile, no evaluative words;
- (iii) adversarially mis-signed adjectives (say "low" where the value is high; numbers unchanged).

## Primary metric and statistics (fixed)

- **Macro-F1 under strict leave-one-subject-out (LOSO)**; per-subject scores always reported.
  Subject is the unit of resampling throughout (n = 15/22/11).
- **Chance**, two definitions, locked in Week 2 before any LLM run: (a) majority-class predictor;
  (b) empirical floor from **1000 within-subject label permutations** re-run through the full LOSO
  pipeline (seed 1337, `configs/classical_floor.yaml`).
- **Uncertainty:** paired-over-subjects bootstrap CIs (1000 resamples, subject as resampling
  unit); **Wilcoxon signed-rank across subjects** for paired condition contrasts.
- **SDS (Surrogate Dependence Score):**
  `SDS = (M_real − M_surrogate) / (M_real − M_chance)`, computed per (dataset × model × paradigm ×
  surrogate) cell, with M = pooled LOSO macro-F1 and M_chance = the permutation floor mean.
- **SDS validity gate (mechanical, coded in the report generator):** if a cell's real-signal
  macro-F1 does not exceed the label-permutation floor by a margin whose 1000-resample bootstrap CI
  excludes zero, **SDS is undefined for that cell**; only the raw delta with CI is reported. No
  ratio is ever printed over a near-zero denominator.
- **P2 margin definition:** margin = M_variant − M_chance; "loses >50%" means
  (margin_i − margin_ii)/margin_i > 0.5, subject to the same validity gate on margin_i.
- **P3 flip rate:** fraction of evaluation items whose argmax label under (iii) differs from (i),
  pooled over LOSO folds; per-subject rates reported.

## Exclusion / missing-data rules (fixed)

- Windows with NaN samples are dropped at windowing time and counted in the manifest.
- Feature values that NeuroKit2 cannot compute for a window are NaN; imputation is median,
  fit within the training fold only.
- No subject is excluded for performance reasons. Subjects failing data-integrity checks
  (corrupt files, missing channels) are excluded before any model run and listed in the manifest.
- LLM inference failures (e.g., OOM) are retried; a cell with any unscored windows is reported
  as incomplete, never silently averaged.

## Analysis code and provenance

All experiments are config-driven; every results directory embeds git SHA, config hash, library
versions, and input-file hashes. Repository: ⟨GitHub/OSF component URL⟩. The full grid is
runnable end-to-end by `scripts/` runners; the report card generator applies the SDS validity gate
mechanically.

## Timeline

Filing precedes all LLM inference (semester Week 3). Grid runs Weeks 4–10; verbalizer ablation
Week 9; write-up Weeks 11–15.
