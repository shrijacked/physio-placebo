# Comparison protocol (Health-LLM PMData stress)

Two tables. A row crosses tables only if the adaptation is the same.

## Table 1 — frozen prompting, MAE, lower is better

| Row | Model | Adaptation | Split | Where the number lives |
|---|---|---|---|---|
| Paper Table 3 | MedAlpaca-7b | Zero-shot | Their seeds {0, 1, 2} | 0.76 ± 0.1 |
| Locked rerun | `medalpaca/medalpaca-7b` | Zero-shot, greedy, first-number scorer | Locked 299-item eval, sha256 `fea3b879…` | `results/anchor/medalpaca/server_run/run.json` (2.689 ± 0.038) |
| Different backbone | `Qwen/Qwen3-8B` | Same locked prompts, greedy, same scorer | Same 299 items, seeds {0, 1, 2} | `results/comparison/prompt_our_models/` |
| Different backbone | `meta-llama/Llama-3.1-8B-Instruct` | Same | Same | Same directory |

The Qwen and Llama rows are the same task and the same prompt text. They are not a
claim that we reproduced MedAlpaca. Instruct models receive that text as the user
turn of their chat template (`enable_thinking` false on Qwen3). MedAlpaca receives
the raw string, as in the locked anchor.

## Table 2 — fine-tuned, MAE, lower is better

| Row | Model | Adaptation | Split | Where the number lives |
|---|---|---|---|---|
| Paper Table 4 | MedAlpaca-7b | Fine-tuned on PMData | Their single-dataset test | 0.38 |
| This arm | `medalpaca/medalpaca-7b` LoRA | `configs/healthllm_finetune.lock.yaml` | Train = released train pool minus locked eval text. Eval = the locked 299, seed 0, same scorer | `results/comparison/medalpaca_ft/` |

Their 10% test split and our locked 299-item eval are not asserted to be the same
split. The caption says so. The fine-tune number is comparable to our 2.689 because
both use that locked eval and that scorer. It is comparable to their 0.38 only as
the same model, same task, same metric, with the split difference written down.

## Train split

`PMData_stress_train_all.json` has 654 rows. The locked eval has 299. One sensor
string is in both. `train_complement` drops it. The written train file has 653
rows, and a test fails if any eval string remains. The tail of their generator's
second half that was never written to either file is not reconstructed.

## Out of both tables

WESAD, MAUS, CogWear, Qwen2.5-VL, SensorLM, ProMind-LLM, and trained non-LLM
WESAD models. Weeks 7–8 surrogates are not part of this arm.

## Results (11 Oct 2026, HTI A6000)

Prompting, three seeds, locked 299-item eval:

- Qwen3-8B: MAE **0.795 ± 0.020** (`results/comparison/prompt_our_models/qwen3_8b/20261010T194851Z_079f00bb/run.json`). Completions are `Answer: 3`.
- Llama-3.1-8B-Instruct: MAE **2654 ± 43** (`results/comparison/prompt_our_models/llama31_8b/20261010T200019Z_3589b7be/run.json`). Completions are Python that paste step counts. The locked scorer reads the first number. It was not retuned.

Fine-tune, eval seed 0:

- MedAlpaca-7b LoRA: MAE **2.697** (297/299 parsed) (`results/comparison/medalpaca_ft/20261010T192207Z_f5228b8c/run.json`). Completions are still `out of 5`. Zero-shot seed 0 on the same eval was 2.714.

## OSF

This arm does not rewrite hypotheses, the surrogate ladder, or the filed Models
section. A local addendum in `docs/osf-prereg.md` says the comparison was added.
Posting it to OSF is a separate step.
