# Week 10 — SDS card

Source of the numbers below: `results/sds_card/20261030T151137Z/card.md`.
Real-signal rows for paradigms A and C are in
`results/eval_grid_rows/20261030T142039Z/` (A6000, commit `2151061`,
`gpu_memory_utilization` 0.80 because other jobs held about 6 GiB). 0 scoring
failures. Every null SDS cell is a ratio or `undefined (validity gate)`.
No paired interval is still pending.
Primary metric is the filed macro-F1. Anything labelled balanced accuracy,
AUROC, matched floor, adjective vote, or input-matched ceiling is
post-registration (`docs/analysis-addendum-2026-10-30.md`). That addendum is
committed and not yet posted to OSF.

## What was run

120 cells: every frozen real-signal setup times the six replacements. Paradigm
C is Qwen2.5-VL on WESAD and MAUS only. Of the 100 null cells, 75 print
`undefined (validity gate)` and 25 print a ratio. The 20 same-activity
permutations are `descriptive, not a null`. Ratios outside [0, 1] are kept.
They are what the formula returns when a replacement scores on the far side of
the locked chance floor.

## Predictions, as filed

**P1 (SDS < 0.3 for raw series and plots).** Unscored wherever the gate fires,
which is both plot cells and most raw-series cells. Where a raw-series ratio
is defined, it is usually not below 0.3. The cell that matters, Qwen on WESAD
with 128 averages of the chest ECG (real macro-F1 0.595), has SDS 0.644 on
Gaussian noise, 0.344 on phase randomization, 0.450 on the wrong person in the
wrong activity, and 1.041 on a flat line and on a deleted signal. Four of the
70 raw-or-plot null cells are below 0.3. P1 does not hold as a general claim.

**P2 (neutralizing the adjectives costs more than half the margin).** Unchanged
from Week 9. Qwen on WESAD loses 17%, so it does not hold there. Qwen on MAUS
loses the whole margin against the filed floor, so it holds there, and only
there: the matched floor for that cell is 0.499 and the original wording is
0.485. CogWear and every Llama original wording do not clear the filed floor,
so P2 is unscored.

**P3 (flipped adjectives change more than 40% of minutes).** Holds for Qwen on
all three datasets (63%, 71%, 72%). Does not hold for Llama (3%, 27%, 8%).

## What the secondary layer changes

Macro-F1 moves when the model changes how often it says the rare class. On
WESAD, always answering A is 0.412 and always answering B is 0.230. Qwen's
WESAD sentences have macro-F1 0.581 and AUROC 0.937, because it says B on 71%
of minutes. The same sentences after phase randomization have macro-F1 0.704
and AUROC 0.872. The higher macro-F1 is a threshold shift. The ordering got
worse. Filed SDS on that rung is −0.732. SDS on AUROC is 0.149, and the
AUROC validity gate passes.

An adjective count with no language model, using the same training-fold
medians, scores WESAD at macro-F1 0.736 and AUROC 0.857. Qwen's margin tracks
that count (Spearman 0.84 on WESAD, 0.88 on MAUS, 0.94 on CogWear).

The 0.918 WESAD logistic regression sees heart, skin, and breath. Paradigms A
and C see one channel. Three summary statistics of the 128 averaged ECG values
score macro-F1 0.786 and AUROC 0.907 on the full roster, and macro-F1 0.805 on
the evaluation subjects. HRV features alone score 0.783 on the full roster.
MAUS PPG and CogWear BVP prompt vectors stay near AUROC 0.5, so those
raw-series cells cannot separate "the model ignored a usable signal" from
"the prompt had no usable signal."

Phase randomization is still a filed null, and it is labelled "keeps level and
spectrum." Correlation with the real WESAD minute is 1.000 for
`eda_level_mean` and `eda_level_std`, 0.998 for `eda_tonic_mean`, and 0.771
for `hrv_mean_hr`.

## Re-score

The second pass of the real minutes agrees with the September numbers. The
largest absolute move in macro-F1 is 0.009 (Qwen, CogWear, every nth sample:
0.576 to 0.585). All 20 real cells are inside 0.02, so none is flagged. The
paired interval for Qwen on WESAD with 128 ECG averages, real minus Gaussian
noise, is 0.116 [0.048, 0.180]. It excludes zero. The filed SDS for that cell
stays 0.644, because SDS uses the locked September score, not the re-score.

## Limitations

Subject is the resampling unit, and CogWear eval is eight people. Identical
prompts still return a few distinct logprobs, so AUROC on a flat line or a
deleted signal is `undefined (no item-level variation)`. Surrogate
input-matched ceilings were not fit: regenerating the windows would overwrite
locked feature files. SensorLM sentences were not run. The Health-LLM
comparison is a separate arm (prompting 0.841 inside 0.76 ± 0.1 on the 14-day
average; full fine-tune 0.442 against their 0.38) and is not an SDS cell.
