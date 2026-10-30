# Analysis addendum — 2026-10-30

**Status.** Post-registration secondary analyses. The primary analysis filed at
https://osf.io/62r5t is unchanged: macro-F1 under strict leave-one-subject-out,
the Surrogate Dependence Score

`SDS = (M_real − M_surrogate) / (M_real − M_chance)`,

and the validity gate (SDS is undefined when the real-signal macro-F1 does not
clear the label-permutation floor by a margin whose bootstrap interval excludes
zero). `M_chance` remains the classical logistic-regression permutation-floor
mean locked in Week 2.

**Motivation.** Weeks 4–9 showed that macro-F1 from the A/B argmax mostly
tracks how often the model says the positive class. A secondary layer is added
so that fact is measured rather than left as a caption. Every number from this
layer is labelled "post-registration (addendum 2026-10-30)" wherever it is
printed.

Posting this file to the OSF registration is a manual step. It is not a rewrite
of the filed Models section, the hypotheses, or the surrogate ladder.

## Seen before this addendum

These numbers were computed in a read-only pass over saved artifacts on
2026-10-30, before this document and before any SDS code. They are disclosed
here so the secondary definitions are not mistaken for a blind analysis.

1. **Macro-F1 from the A/B argmax mostly measures label bias.** Chance macro-F1
   depends on how often the model says B. Always-A on WESAD is 0.412, always-B
   is 0.230, and a label-independent 50/50 guess is about 0.48. The filed
   chance (classical permutation mean, WESAD 0.419) does not match LLM answer
   rates.
   - Qwen WESAD A `bin_mean` with Gaussian noise: 0.482 [0.438, 0.525], above
     the "0.41 guessing" line. Its matched chance is 0.499.
   - Qwen WESAD B Gaussian: 0.292 (answers B 92% of the time). Filed SDS for
     that rung would be about 1.77, outside [0, 1].
2. **Saved logprobs carry far more than argmax shows.** AUROC of the margin
   `logp(B) − logp(A)`.
   - Qwen WESAD B (original words): macro-F1 0.581, AUROC 0.937, answers B on
     71% of minutes (true rate about 30%).
   - Llama WESAD B: macro-F1 0.278, AUROC 0.819.
   - Qwen2.5-VL WESAD C, same-activity-other-person rung: macro-F1 0.652,
     AUROC 0.794. The real-signal C rows were never saved, so "plots are at
     chance" is unverified beyond argmax.
4. **Adjective-vote baseline (no LLM).** Counting stress-typical minus
   calmer-typical words, using the same training-fold medians, scores WESAD
   (i) at macro-F1 0.736, balanced accuracy 0.778, AUROC 0.857, which beats
   Qwen's 0.581. Spearman(Qwen margin, vote) is 0.84 on WESAD and 0.88–0.94
   on MAUS and CogWear.
5. **IAAFT is not an empty null for these features.** Per-window IAAFT keeps
   the value distribution and the spectrum. Correlation of the real window
   with its IAAFT surrogate: `eda_level_mean` 1.000, `eda_level_std` 1.000,
   `eda_tonic_mean` 0.998, `resp_amp_mean` 0.933, `hrv_mean_hr` 0.771.
6. **The headline comparison is not input-matched.** Paradigms A and C show
   one channel (WESAD chest ECG, MAUS PPG, CogWear BVP). HRV-only logistic
   regression on WESAD is 0.783, not the 18-feature 0.918. A logistic
   regression on three summary statistics (mean, standard deviation, mean
   absolute difference) of the exact 128 `bin_mean` ECG values scores F1
   0.786 and AUROC 0.907. MAUS PPG and CogWear BVP prompt vectors score about
   AUROC 0.5.
7. **Numeric noise.** Identical prompts (signal deleted, flat line) return
   2–4 distinct logprobs (standard deviation 0.002–0.08). Rows are scored in
   subject-then-condition order, so that noise is ordered with the label, and
   AUROC on near-constant cells is spurious (up to 0.70). Re-scoring Paradigm
   B moved Qwen WESAD from 0.584 to 0.581.
8. **Week 9 prediction P2 on MAUS.** Qwen MAUS variant (i) is 0.485, against a
   matched floor of 0.499 (95th percentile 0.519). The prediction "holds" on
   MAUS only under the filed classical floor. The card reports that caveat.

Finding 3 (real-signal A and C rows were not saved) and findings 9–10 (report
wording and repo hygiene) are engineering facts, not analysis numbers. They
are in `PROGRESS.md` (2026-10-30).

## Definitions used by the secondary layer

Code that prints these quantities uses the same definitions, in the same
words, in its docstrings.

- **Margin.** `m = logprobs["B"] − logprobs["A"]`. Token B is the positive
  class (`y = 1`). Token A is the negative class (`y = 0`).
- **Balanced accuracy (BA).** Mean of the two class hit rates. For any
  predictor whose answer does not depend on the label, BA is exactly 0.5.
- **Pooled AUROC.** AUROC of `m` against `y`, pooled over evaluation items.
  The 95% interval resamples subjects with replacement, 1000 draws, seed
  1337, and recomputes the pooled AUROC on each draw.
- **AUROC item-variation gate.** AUROC is defined only when
  `std(m) > 10 * max(std(m in the deleted cell of the same model, dataset,
  paradigm, and scheme), 0.01)`. Otherwise the cell is
  "undefined (no item-level variation)".
- **Matched permutation floor.** Shuffle `y` within each subject, 1000 times,
  seed 1337, leaving `yhat` fixed. Report the mean and the 95th percentile of
  the resulting macro-F1 values.
- **SDS on balanced accuracy.**
  `SDS_BA = (BA_real − BA_surr) / (BA_real − 0.5)`. Defined only when the
  subject-bootstrap interval for `BA_real` has lower bound strictly above 0.5.
- **SDS on AUROC.**
  `SDS_AUC = (AUC_real − AUC_surr) / (AUC_real − 0.5)`. Defined only when the
  subject-bootstrap interval for `AUC_real` has lower bound strictly above
  0.5, and the item-variation gate passes for both the real cell and the
  surrogate cell.
- **Matched-floor variant of the filed gate.** The locked real-signal
  `ci_lo` is above the matched permutation-floor mean. This does not replace
  the filed gate. It is reported beside it.
- **Holm.** Holm adjustment across every null-rung Wilcoxon signed-rank
  contrast in the card. `subject_perm_preserved` is descriptive and is not in
  that family.
- **Adjective-vote baseline.** For each feature, compare the value with the
  training-fold median (everyone except the person being scored, including
  the prompt-development subjects). A stress-high feature contributes +1 when
  it is above the median and −1 when it is below. A calm-high feature
  contributes the opposite. Other features and missing values contribute 0.
  The prediction is B when the sum is positive and A otherwise, including
  ties. Variant (iii) negates the vote. The same rule is applied to surrogate
  feature matrices. Reported next to every Paradigm B cell that has saved
  rows: macro-F1, BA, AUROC of the vote, agreement with the model's argmax,
  and Spearman correlation of the vote with the model's margin.
- **Input-matched ceilings.** Logistic regression, the locked Week-2 pipeline
  (median impute, z-score, L2, `C = 1`, seed 1337), strict leave-one-subject-out,
  on the real windows only. Two inputs. (1) Three summary statistics of the
  exact Paradigm A prompt vector (mean, standard deviation, mean absolute
  difference), both downsample schemes, all three datasets, primary channel
  only, scored on the evaluation subjects. (2) The `hrv_`-prefixed columns of
  the locked feature matrix, one ceiling per dataset. Surrogate windows are
  not on this machine, and regenerating them would overwrite locked feature
  files, so surrogate input-matched ceilings are not in this card.
- **IAAFT label.** IAAFT stays a null rung in the filed SDS table. Every
  IAAFT row is also labelled "keeps level and spectrum (see ledger)".

## What this addendum does not change

Predictions P1, P2, and P3 are scored with the filed definitions. A cell whose
filed SDS is undefined stays undefined in the primary table. A ratio outside
[0, 1] is printed as computed. It is not clipped.
