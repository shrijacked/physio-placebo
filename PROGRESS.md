# PROGRESS

## Current status
- Semester week: **Weeks 4–5 CLOSED** (2026-09-21). Tag `week-05-done`.
  Week 6 is next and **not started**.
- Phase: real-signal Paradigms A/B are in for both open models. Surrogates
  wait for Weeks 7–8. Plots wait for Week 6.
- Last session: 2026-09-21 — eval grid finished on the HTI A6000
  (`eval_vllm_qwen3_8b.json` 23:52 IST, `eval_vllm_llama31_8b.json` 00:13 IST).
  18 cells, 11,916 scored windows, 0 scoring failures.

## Week checklist
- [x] Week 1 — data + harness foundation (**WESAD 15, MAUS 22, CogWear 10 usable / 11**)
  - [x] WESAD downloaded (2.25 GB zip, sha256 recorded), windowed end-to-end: 15/15 subjects,
        1,042 windows (60 s / 30 s stride), 731 non-stress / 311 stress, 0 NaN drops
        (`data/processed/wesad/prepare_manifest.json`)
  - [x] Frozen NeuroKit2 feature extraction (8 HRV + 7 EDA/cvxEDA + 3 RESP): all 1,042 WESAD
        windows, **zero NaNs in all 18 features**
        (`results/features/wesad/20260827T123554Z_84ac4582/`, parquet sha256 `e312262c…`)
  - [x] Feature code frozen: `configs/frozen_features.lock.json`
        (source sha256 `702d5f37…`, neurokit2 0.2.13, freeze-guard test active)
  - [x] Strict LOSO harness + leakage tests + prompt-dev split (`src/physio_placebo/data/loso.py`)
  - [x] MAUS loader written per authors' baseline-repo format; fixture-tested; **real data
        ingested 2026-09-16** (IEEE DataPort `MAUS.zip`, 22/22 subjects in
        `SUBJECT_IDS`, concat sha256 of all `pixart.csv` `fadf7131…`)
  - [x] Health-LLM code + paper read; `docs/health-llm-notes.md` written (verified line-level
        code defects; headline table verified from paper PDF; majority-class MAE ≈ 0.434
        computed from their own Table 16 — beats most of their LLM cells)
  - [x] SensorLM `captioning.py`/`constants.py` vendored (commit `7d087e9`, Apache-2.0 per
        upstream README), excluded from lint, **templates render on a real WESAD window**
        (4 passing tests in `tests/test_sensorlm_render.py`)
  - [x] CogWear loader + windowing (pilot cohort, baseline=0 vs Stroop=1; E4 BVP@64Hz +
        EDA@4Hz overlap-trimmed; clock validation), written against the published
        SHA256SUMS.txt/README/sample files; 6 fixture tests pass. **Upstream gap:**
        `pilot/3/cognitive_load/empatica_eda.csv` does not exist on PhysioNet → subject 3
        excluded by the missing-channels rule → **10 usable of 11** (recorded in prereg + manifest)
  - [x] CogWear real data downloaded via per-file HTTPS (43/43 CSVs, each SHA256-verified
        against the published SHA256SUMS.txt; `scripts/download_cogwear.py` — the monolithic
        zip kept resetting and retries were served non-resumable). The abandoned zip resume
        loop eventually completed too (2026-08-27 23:07 IST): the full official archive is
        also on disk at `data/raw/cogwear/cogwear-1.0.0.zip` (187,856,113 bytes, `unzip -t`
        clean, sha256 sidecar) — kept as canonical backup; processed windows remain built
        from the per-file-verified CSVs. Windowed end-to-end:
        **10 subjects, 114 windows (39 baseline / 75 Stroop), 0 NaN drops**
        (`data/processed/cogwear/prepare_manifest.json`)
  - [x] CogWear features: all 114 windows, zero NaNs in all 15 features
        (`results/features/cogwear/20260827T134111Z_d5b1bd65/`)
  - [x] MAUS raw data: IEEE DataPort zip dropped 2026-09-16 into `data/raw/maus/MAUS/`
        (`Data/Raw_data/<id>/pixart.csv` × 22). Windowed: **22 subjects, 1,086 windows**
        (362 low / 724 high), **0 NaN drops** (`data/processed/maus/prepare_manifest.json`).
        Subjects 002–006: 54 windows each (100 Hz); 008–025: 48 each (102.5 Hz).
  - [x] MAUS features: all 1,086 windows, **zero NaNs in all 8 HRV features**
        (`results/features/maus/20260916T131937Z_3aa179ba/`, parquet sha256 `e9367ac1…`)
- [x] Week 2 — classical floor (**WESAD + MAUS + CogWear locked**)
  - [x] WESAD floor locked before any LLM run
        (`results/classical_floor/wesad/20260827T130307Z_600c388f/floor.json`, seed 1337):
        logreg L2 pooled macro-F1 **0.9181** (per-subject 0.9127 ± 0.1066);
        depth-3 tree **0.8477** (0.8456 ± 0.1286);
        majority-class **0.4123**; permutation floor (n=1000, within-subject):
        logreg mean 0.4193 / p95 0.4327, tree mean 0.4388 / p95 0.4690
  - [x] CogWear floor locked
        (`results/classical_floor/cogwear/20260827T134224Z_842011b7/floor.json`, seed 1337):
        logreg L2 pooled macro-F1 **0.5692** (per-subject 0.5069 ± 0.2275);
        depth-3 tree **0.5598** (0.4986 ± 0.1649); majority **0.3968**;
        permutation floor: logreg mean 0.4681 / **p95 0.5412**, tree mean 0.4783 / p95 0.5656.
        **Note:** classical signal is weak — logreg barely clears the permutation p95 and the
        tree does not; CogWear cells are likely to trigger the pre-registered SDS validity
        gate. WESAD is the strong-signal dataset (0.92 vs ~0.42).
  - [x] MAUS floor locked
        (`results/classical_floor/maus/20260916T132700Z_4ded037e/floor.json`, seed 1337):
        logreg L2 pooled macro-F1 **0.4440** (per-subject 0.4370 ± 0.0541);
        depth-3 tree **0.4815** (0.4536 ± 0.0864); majority **0.4000**;
        permutation floor: logreg mean 0.4027 / p95 0.4129, tree mean 0.4244 / p95 0.4534.
        **Note:** classical signal is weak (tree just clears p95; logreg also, but barely).
        MAUS cells are likely to trigger the SDS validity gate, same family as CogWear.
        Byte-identity now covers MAUS (`configs/locked_floors.yaml`).
  - [x] Byte-identity test (baselines vs LLM feature matrices) —
        `tests/test_byte_identity.py` + `physio_placebo.features.matrix`. Locked
        WESAD/CogWear/MAUS parquet SHA-256s match `floor.json`. Week-4 verbalizer must
        call `load_locked_feature_matrix` / `assert_matches_locked_floor`.
- [x] Week 3 — fidelity anchor + OSF pre-registration (**CLOSED 2026-09-20**)
  - [x] Anchor chosen + recipe + patch list: `docs/fidelity-anchor.md` (PMData stress,
        zero-shot MedAlpaca-7b, target MAE 0.76 ± 0.1)
  - [x] Fallback pipeline prepared (eval split locked, patches, scorer). **API run abandoned
        2026-09-16** after 35/897 calls at the 50 req/day cap; support confirmed credits do
        not lift limits. Do not wait on this cell.
  - [x] OSF prereg **FILED & PUBLIC 2026-08-30 18:56 IST**: registration
        https://osf.io/62r5t (DOI 10.17605/OSF.IO/62R5T, Open-Ended Registration,
        OSF Registries; associated project https://osf.io/arj8w). Filed text =
        `docs/osf-prereg.md` @ `f705db9`. Verified publicly visible while logged out,
        2026-08-30 19:10 IST. **The no-LLM-before-prereg gate is CLEARED.**
  - [x] Zero-shot prompt construction locked (2026-09-16 dry-run): 299 items × seeds
        {0,1,2}, eval sha256 `fea3b879…`, prompt sha256s `0abba6a2` / `6789c809` /
        `50bee721` under `results/anchor/medalpaca/20260916T150339Z_29ef1798/`.
        `tests/test_anchor.py` 9 passing. P2 (`medalpaca_pl`) is in
        `scripts/run_medalpaca_anchor.py`.
  - [x] OSF model-deviation update public 2026-09-20 on https://osf.io/62r5t
        (submitted 15:02 UTC, approved; schema response `6aaff4804f661ffb39d0ab73`).
        Reason-for-update text is the locked deviation paragraph.
  - [x] Anchor *number* (2026-09-20, `10.1.45.49` RTX A6000): MAE **2.689 ± 0.038**
        (seeds 2.714 / 2.645 / 2.708), parse failures 6/897, gap **+1.929** vs
        0.76 ± 0.1. Prompts matched the 2026-09-16 lock. Completions are mostly
        `out of 5` / `out of 10`; first-number scoring reads 5 or 10. Documented
        in `docs/fidelity-anchor.md` §6. Do not retune the scorer.
- [x] Weeks 4–5 — Paradigms A and B, prompt sweep then freeze (**CLOSED 2026-09-21**)
  - [x] Logprob-over-label-tokens scorer + known-logprob fixture
        (`src/physio_placebo/scoring/logprob.py`, `tests/test_logprob_scorer.py`)
  - [x] Label tokens locked as A/B (`configs/label_tokens.yaml`)
  - [x] Three prompt templates × {0-shot, 4-shot} renderer
  - [x] Verbalizer variant (i) on locked features, train-fold medians
  - [x] Two Paradigm A downsample schemes documented + tested
        (`uniform_stride` n=256, `bin_mean` n=128)
  - [x] vLLM client + extract helpers (unit-tested without GPU)
  - [x] Prompt-dev sweep runner (`scripts/run_prompt_sweep.py`)
  - [x] Real vLLM prompt-dev sweeps on the A6000 (Qwen3-8B + Llama-3.1-8B-Instruct)
  - [x] Frozen prompt IDs (`configs/frozen_prompts.lock.yaml`, 18 cells)
  - [x] Eval-grid runner + subject-bootstrap CIs (unit-tested, ConstantClient)
  - [x] Real-signal A/B × 2 models × 3 datasets with subject-bootstrap CIs
        (`results/eval_grid/eval_vllm_qwen3_8b.json`,
        `results/eval_grid/eval_vllm_llama31_8b.json`). Eval subjects only;
        prompt-dev held out (wesad S2/S8, maus 015/023, cogwear 4/8).
        5,958 windows × 2 models, 0 failures. Strongest cell: Qwen3-8B
        WESAD A `bin_mean` / clinical 0-shot **0.595 [0.550, 0.647]**.
        Llama MAUS A `bin_mean` and MAUS B collapsed to **0.400 [0.400, 0.400]**
        (constant-class on every eval subject). CogWear CIs are wide (n=90).

  | dataset | cell | Qwen3-8B | Llama-3.1-8B |
  |---|---|---|---|
  | wesad | A stride | 0.427 [0.392, 0.465] | 0.333 [0.301, 0.370] |
  | wesad | A bin_mean | **0.595 [0.550, 0.647]** | 0.230 [0.227, 0.233] |
  | wesad | B | 0.584 [0.462, 0.699] | 0.283 [0.247, 0.324] |
  | maus | A stride | 0.467 [0.436, 0.499] | 0.413 [0.403, 0.426] |
  | maus | A bin_mean | 0.483 [0.452, 0.516] | 0.400 [0.400, 0.400] |
  | maus | B | 0.487 [0.439, 0.531] | 0.400 [0.400, 0.400] |
  | cogwear | A stride | 0.576 [0.443, 0.689] | 0.404 [0.392, 0.417] |
  | cogwear | A bin_mean | 0.396 [0.380, 0.411] | 0.440 [0.399, 0.495] |
  | cogwear | B | 0.475 [0.393, 0.504] | 0.479 [0.375, 0.640] |

- [ ] Weeks 6–15: not started

## Open blockers
- ~~**GPU hardware**~~ **RESOLVED 2026-09-20**: account `shrijak` on Plaksha HTI
  `10.1.45.49`, RTX A6000 48 GB. Used for the Week-3 anchor and the Weeks 4–5
  sweep + eval grid. This Mac still cannot run the grid.
- ~~OpenAI account tier~~ **CLOSED 2026-09-16 as a grid path.** Credits (~$2,499) sit in
  `codex-btzye1` with a 50 req/day/model cap; support (Kristian, 2026-09-10) confirmed
  credits do not lift rate limits. Stretch GPT cell stays optional and unused.
- ~~MAUS raw data~~ **RESOLVED 2026-09-16**: IEEE DataPort `MAUS/` (22 pixart.csv,
  concat sha256 `fadf7131…`) at `data/raw/maus/MAUS/Data/Raw_data`. Windowed 1,086
  windows, 0 NaN drops; features parquet `e9367ac1…`.
- ~~CogWear download~~ **RESOLVED 2026-08-27** via per-file downloads with SHA256 verification
  (`scripts/download_cogwear.py`); the zip resume-loop was abandoned. (This line was stale.)
- ~~PMData~~ **RESOLVED 2026-08-27**: `data/raw/pmdata/pmdata.zip` downloaded complete
  (1,416,129,266 bytes, 912 files, sha256 `53a49d94…` in sidecar). Anchor data side is ready;
  the primary (MedAlpaca) run completed 2026-09-20. GPT-3.5 fallback abandoned 2026-09-16.
- **Prof sign-off** — verbalizer variant (iv) (SensorLM templates) still pending
  (PROJECT_PLAN.md §11); vendoring done regardless as planned.
- ~~**OSF model deviation**~~ **RESOLVED 2026-09-20**: public registration update on
  https://osf.io/62r5t (approved). Filed Models section still names gpt-4.1; executed
  grid is Qwen/Llama/vLLM. Local addendum in `docs/osf-prereg.md`.
- ~~OSF filing~~ **RESOLVED 2026-08-30**: registered and public at https://osf.io/62r5t
  (DOI 10.17605/OSF.IO/62R5T). Filed text = `docs/osf-prereg.md` @ `f705db9`.
- ~~MAUS access~~ **RESOLVED 2026-09-16** via IEEE DataPort download (user).

## Decisions log
- 2026-09-21: **Weeks 4–5 closed.** Real-signal A/B eval grid on the HTI A6000
  (fp16 vLLM 0.29, `max_model_len` 32768, FlashInfer sampler off). Artifacts
  `results/eval_grid/eval_vllm_{qwen3_8b,llama31_8b}.json`. Do not retune
  frozen prompts. Do not start Week 6 until asked.
- 2026-09-20: **Prompt IDs frozen** from `sweep_vllm_qwen3_8b.json` and
  `sweep_vllm_llama31_8b.json` (4608 scored / 0 failures each). Lockfile
  `configs/frozen_prompts.lock.yaml`. Do not retune. Eval grid is next.
- 2026-09-20: **vLLM 0.29 dropped bitsandbytes.** Prompt-dev sweep uses the same
  HF ids (`Qwen/Qwen3-8B`, later Llama-3.1-8B-Instruct) in fp16 on the A6000
  48GB. OSF Models text not rewritten. 4-bit was a 16GB-card constraint.
- 2026-09-20: Prompt-dev sweep builds Paradigm A bodies for prompt-dev subjects
  only (`n_dev=2`, seed 1337). Split is still computed on the full locked roster.
- 2026-09-20: **Week 4 opened.** First slice is the logprob scorer, A/B tokens,
  three templates, verbalizer (i), and two downsample schemes. No surrogate
  code. No Week 6 plots. GPU grid waits on the prompt sweep.
- 2026-09-20: OSF model-deviation **update verified public** on https://osf.io/62r5t
  (submitted 15:02 UTC, approved). Week 3 closed.
- 2026-09-20: **Fidelity-anchor number obtained.** Zero-shot MedAlpaca-7b on the
  HTI A6000: MAE 2.689 ± 0.038 vs paper 0.76 ± 0.1 (gap +1.929). Report the gap;
  do not change the locked first-number scorer.
- 2026-09-18: Supervisor briefing written for Dr.\ Siddharth
  (`reports/supervisor_briefing_2026-09-18.tex` / `.pdf`): Weeks 1--2 locked
  numbers, OSF URL, three asks (GPU ≥16 GB, SensorLM variant iv, Week-10 format).
- 2026-09-16: **Week 3 opened.** Hardware re-check: M2 Pro, 16 GB unified, 14 GiB free
  disk, no CUDA — MedAlpaca fp16 still impossible here. Zero-shot prompts locked
  without loading weights. Week 3 is not closed.
- 2026-09-16: **Week 2 closed.** MAUS floor locked (tree 0.4815 vs perm p95 0.4534;
  logreg 0.4440 vs p95 0.4129; majority 0.4000). Weak classical signal — SDS validity
  gate likely. Byte-identity registry now includes MAUS.
- 2026-09-16: **Week 1 closed.** MAUS real data ingested (22/22, 1086 windows, 0 feature
  NaNs). Next week is Week 2 (MAUS classical floor only; WESAD/CogWear already locked).
- 2026-09-16: Week-2 byte-identity contract locked (`configs/locked_floors.yaml`,
  `features.matrix`, 6 tests). MAUS still has no raw files under `data/raw/maus/`;
  the IEEE login remains a user action. Fixture-level MAUS prepare→features→floor
  now has a regression test so the real download is mechanical.
- 2026-09-16: **OpenAI API grid abandoned; original vLLM stack restored.** Support (Kristian)
  confirmed ~$2,499 credits remain but do not remove 50 req/day limits. Grid models:
  Qwen3-8B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-VL-7B-Instruct, 4-bit, logprob scoring.
  Filed OSF still names gpt-4.1 — local addendum in `docs/osf-prereg.md`; user must post the
  same comment on osf.io/62r5t. (User-directed.)
- 2026-08-30 (evening): **Fallback anchor executed up to the API wall.** Their pipeline needed
  3 more patches than planned, all discovered by execution and logged as notes §3.11–13:
  the released generator and inference scripts use incompatible eval schemas (adapter added);
  `set_seed()` is never called (patched in as intended); p12/p13 lack `resting_heart_rate.json`
  upstream and the released code bleeds the previous participant's sensors into their rows
  (patched to a clean skip → 953 usable items vs the paper's implied 1,418; disclosed).
  Eval split locked: 299 items, label dist {1:4, 2:53, 3:186, 4:53, 5:3}, constant-3 MAE
  0.401 on this set. Run died at OpenAI's free-tier 50 req/day cap after 35 items; runner made
  checkpoint/resumable (patch I9); the 35 finished calls were NOT salvaged from the log (tqdm
  carriage returns corrupted the stream; worth $0.05 — clean rerun preferred).
- 2026-08-30: **Grid models swapped** from open-weights (Qwen3-8B, Llama-3.1-8B, Qwen2.5-VL via
  vLLM) to OpenAI dated snapshots **`gpt-4.1-2025-04-14` + `gpt-4.1-mini-2025-04-14`** (vision
  paradigm C via the flagship). Drivers: user holds an OpenAI key, no GPU available, and the
  audited literature (Health-LLM) used OpenAI models — ecological validity. gpt-5.x snapshots
  rejected empirically: no logprob support ("Unsupported parameter", verified 2026-08-30);
  gpt-4.1 pair returns top-20 logprobs with both label tokens present. Logprob scoring rule
  unchanged. Open-weights 8B kept as optional extension if GPU materializes. (User-directed.)
- 2026-08-30: OpenAI key stored in git-ignored `.env` (user-authorized for this project;
  `.gitignore` extended with `.env` — it previously only covered `.venv/`). Key should be
  rotated at semester end since it transited chat.
- 2026-08-30: Anchor fallback is now **viable**: `gpt-3.5-turbo-instruct` (their few-shot
  0.94 ± 0.1 cell) is still served and we now hold a key — both conditions from
  `docs/fidelity-anchor.md` met. MedAlpaca zero-shot remains the primary (GPU-gated).
- 2026-08-27: Skipped gstack office-hours/autoplan for this session — `CURSOR_SEMESTER_AGENT.md`
  + `PROJECT_PLAN.md` are the already-approved plan; user said "get started". (User-directed.)
- 2026-08-27: WESAD windowing 60 s / 30 s stride, stress={2}, non-stress={1,3} (standard WESAD
  binary task; windows cut strictly inside one protocol segment). Config-hashed in
  `configs/windowing.yaml`.
- 2026-08-27: Permutation floor scheme = within-subject label shuffles, full LOSO re-run per
  permutation, n=1000, seed 1337 (`configs/classical_floor.yaml`).
- 2026-08-27: Fidelity anchor = PMData stress, **zero-shot MedAlpaca-7b (MAE 0.76 ± 0.1)**;
  fallback few-shot GPT-3.5 (0.94). Rationale: open weights; their zero-shot GPT-3.5/4 stress
  cells failed ("—"); Gemini-Pro 1.0 retired; zero-shot avoids their few-shot leakage.
  (Agent decision, reversible until prereg filing — flag if you disagree.)
- 2026-08-27: CogWear fetched from PhysioNet **static** zip URL (the on-the-fly
  `get-zip` endpoint reset repeatedly).
- 2026-08-27: Feature code frozen at git `ffb81a7`, `frozen_py_sha256=702d5f37…`.
- 2026-08-27: SensorLM vendored at upstream commit `7d087e9`; Apache-2.0 grant lives in
  upstream README (no root LICENSE file exists upstream); vendor dir excluded from ruff.

## NOT IMPLEMENTED / known gaps
- ~~CogWear loader/windowing/floor~~ RESOLVED 2026-08-27 (per-file download; loader, windows,
  and floor all landed — this line was stale).
- MAUS on real data — **RESOLVED 2026-09-16**.
- ~~Fidelity-anchor reproduction *number*~~ **RESOLVED 2026-09-20**: MAE 2.689 ± 0.038,
  gap +1.929, artifact `results/anchor/medalpaca/server_run/`.
- Everything LLM-side after the real-signal A/B grid: Paradigm C (Week 6),
  surrogates (Weeks 7–8), verbalizer ablation (Week 9). Weeks 4–5 A/B numbers
  are in `results/eval_grid/`.
- `pip install -e .` required `required_permissions=all` in this sandbox (a `.pth` write);
  fresh-clone install on a second machine still unverified (GPU box required again).
- Health-LLM releases **no evaluation code** (verified 2026-08-27: no `eval/` dir;
  `medalpaca/` = training/inference utilities only). The paper's MAE must be re-implemented
  for the anchor; convention fixed in `docs/fidelity-anchor.md` §2 and disclosed as part of
  the reproduction gap.
