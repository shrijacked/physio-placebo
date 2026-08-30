> **Registered:** https://osf.io/62r5t — DOI
> [10.17605/OSF.IO/62R5T](https://doi.org/10.17605/OSF.IO/62R5T), Open-Ended Registration,
> registered 2026-08-30 18:56 IST, public. The text below is the filed version (commit
> `f705db9`). Any post-filing protocol change is a deviation and must be reported as such.

## Title

Do large language models read the signal or the story? A surrogate-ablation audit of LLM
"physiological reasoning" on wearable sensor data.

## Authors

Shrijak Kumar. Supervisor: Dr. Siddharth.

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

## Hypotheses (directional, falsifiable)

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

Not applicable (no participants). To guard against analyst degrees of freedom, all conditions,
metrics, exclusion rules, and the analysis pipeline are fixed here before any LLM inference is run.

## Existing data

At the time of filing:

- All datasets are public, previously published (WESAD, MAUS, CogWear).
- Data preparation (windowing), feature extraction (frozen, source hash `702d5f37…`,
  `configs/frozen_features.lock.json`), and the **classical baselines/chance floors have
  already been computed for WESAD and CogWear** (`results/classical_floor/`, locked before this
  filing); the MAUS floor will be locked the same way, with the identical frozen pipeline, before
  any LLM run on MAUS. Observed floors: WESAD logreg macro-F1 0.918 vs permutation chance ≈ 0.42;
  CogWear logreg 0.569 vs permutation p95 0.541 (weak classical signal — the SDS validity gate is
  expected to bind for many CogWear cells).
- **No LLM inference of any kind has been run on any dataset** at filing time. The LLM
  experimental grid — the subject of this pre-registration — is untouched.
- One reproduction of a published Health-LLM number (fidelity anchor, on the original authors'
  PMData stress task) has **not** been run at filing time; it uses their data and task and
  shares no data or code path with our grid.

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

NeuroKit2 v0.2.13 extraction; code frozen and hash-locked prior to this filing (source hash
recorded in `configs/frozen_features.lock.json`): 8 HRV time/frequency features, 7 EDA features
(cvxEDA tonic/phasic), 3 respiration features (dataset-dependent availability). Classical
baselines and the LLM verbalizer consume **byte-identical** feature matrices (enforced by test).

## Models (fixed)

- OpenAI `gpt-4.1-2025-04-14` (flagship) and `gpt-4.1-mini-2025-04-14` (mini tier), accessed
  through the Chat Completions API with dated snapshot pins, temperature 0, fixed seed; the
  returned `system_fingerprint` is logged for every call. These snapshots are selected because
  they expose token log-probabilities (verified 2026-08-30); current reasoning-tuned snapshots
  (gpt-5.x) do not, and are therefore incompatible with the scoring rule below.
- Paradigm C uses `gpt-4.1-2025-04-14` image input (WESAD and MAUS only).
- **Scoring: log-probabilities of the label tokens, read from the top-20 log-probabilities at
  the single answer position (`max_completion_tokens = 1`; labels are single tokens by
  construction).** The prediction is the argmax over the label-token log-probabilities. No free
  generation, no answer parsing. Items where neither label token appears in the top-20 are
  counted and reported as scoring failures, never imputed.
- Disclosed limitation: these are closed-weight hosted snapshots that the provider may
  eventually deprecate. All request/response payloads (including logprobs) are archived, so
  every scored output remains re-analyzable indefinitely.
- Budget rule (fixed in advance): the mini model runs the full grid; the flagship runs the full
  grid if projected cost is within budget (US$150), otherwise a pre-specified subset (paradigm
  B on all datasets; paradigms A and C on WESAD only).
- Optional extension (only if GPU access materializes): one open-weights 8B instruct model
  served locally with vLLM, scored with the same rule over full-vocabulary log-probabilities.

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
- **Chance**, two definitions, locked before any LLM run: (a) majority-class predictor;
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
versions, and input-file hashes. Repository: https://github.com/shrijacked/physio-placebo. The
full grid is runnable end-to-end by `scripts/` runners; the report card generator applies the SDS
validity gate mechanically.

## Execution order (fixed)

1. This registration is filed before any LLM inference is run.
2. The MAUS classical floor is locked upon data receipt, using the identical frozen pipeline,
   before any LLM run on MAUS.
3. Prompt templates are selected on the prompt-development fold and frozen.
4. Real-signal grid: all paradigms × models × datasets.
5. Surrogate ladder across the full grid, with the information-preservation ledger.
6. Verbalizer-leakage ablation.
7. Pre-registered analyses and write-up. Any deviation from this document will be reported as
   such.
