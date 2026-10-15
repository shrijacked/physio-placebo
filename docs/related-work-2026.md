# Related work for the comparison arm (2023–2026)

Window: January 2023 through October 2026. Venue bar: CHIL, NeurIPS, ICLR, ICML,
ACL/EMNLP, CHI, IMWUT/UbiComp, AAAI, ACII, ICMI. A number is comparable only when
dataset, task, model, adaptation (prompt vs fine-tune), split, and metric all agree.

## Matrix

| Paper | Venue | Dataset | Signals | Model | Adaptation | Split | Metric | Verdict |
|---|---|---|---|---|---|---|---|---|
| Kim et al., Health-LLM, arXiv 2401.06866 | CHIL 2024 | PMData (we have it), plus LifeSnaps, GLOBEM, AW_FB | Daily steps, calories, resting heart rate, sleep minutes, mood | MedAlpaca-7b (open). HealthAlpaca is their fine-tune. GPT-4 / Gemini are closed. | Zero-shot and few-shot **and** instruction fine-tune, reported separately | Record shuffle, seed 123 in the released generator. Paper text says a 10% test split. Not LOSO. | Stress MAE | **Match.** Two tables, not one. |
| Zhang et al., SensorLM, arXiv 2506.09108 | NeurIPS 2025 | Private Fitbit/Pixel, 59.7M hours, 103,643 people | PPG, ACC, EDA, temp, altimeter, minute-level | Sensor encoder + CLIP/CoCa. No public weights. Not Qwen or Llama. | Trained from scratch | Their private split | Activity / metabolic, not PMData stress MAE | Cite only. |
| Zheng et al., ProMind-LLM, arXiv 2505.14038 | No conference on the arXiv record | PMData and GLOBEM | Fitbit behavior plus self-reports | Llama3-8B and InternLM2-7B, continued pretraining + SFT | Prompt and PT+SFT in one table, already separated | Their psychologist-made binary risk labels. Not in the public PMData release. | Binary F1 | Out until a venue and a label file exist. |
| Soyebo et al., arXiv 2609.11961 | Preprint, Sep 2026 | PMData | Same daily summaries as Health-LLM | Qwen3-0.6B / 1.7B, Granite-3.3-2B | Zero-shot only | Not LOSO | MAE | Supporting note. Wrong venue, smaller models. |
| Hagos et al., arXiv 2609.20991 | Under review | WESAD, EmoWear | Raw wrist and chest signals, 2.5 s windows | LSTM, TCN, Transformer | Trained classifiers | LOSO | 3-class accuracy | Not an LLM paper. Not in the LLM tables. |

No 2023–2026 conference paper read for this note fine-tunes Llama-3.1 or Qwen3 on WESAD.
An arXiv query for WESAD and Llama returned no hits (11 Oct 2026). The WESAD /
MAUS / CogWear macro-F1 grid stays our own experiment.

## Health-LLM, the paper we match

Read from the arXiv HTML (v2) and from the released repo at
`configs/data/third_party/Health-LLM`.

**Prompting (Table 3).** PMData stress, zero-shot MedAlpaca-7b, MAE **0.76 ± 0.1**
over seeds {0, 1, 2}. Our locked rerun is MAE **2.689 ± 0.038**
(`results/anchor/medalpaca/server_run/run.json`). The first-number scorer was not
changed.

**Fine-tune (Table 4 and Appendix A).** MedAlpaca-7b fine-tuned on PMData and
tested on PMData stress: MAE **0.38**. Appendix A: 7B gets 3 epochs, 13B gets 5,
batch 128, Adam, learning rate 2e-5, cosine schedule, warmup 100, cross-entropy.
Further knobs follow `kbressem/medAlpaca` defaults (LoRA r=8, alpha=16, dropout
0.1, `q_proj` and `v_proj`). The released `finetune.sh` is a different run: full
fine-tune, `use_lora` false, `llama-13b-bf`, 5 epochs, 4 GPUs. We do not run that
script. The lock file is `configs/healthllm_finetune.lock.yaml`.

Official HealthAlpaca weights were not on Hugging Face. Unofficial
`xw17/HealthAlpaca-*-lora` uploads (April 2026, zero downloads) are not loaded.

## What this note does not do

It does not put WESAD macro-F1 0.595 next to PMData MAE. It does not start the
surrogate ladder.
