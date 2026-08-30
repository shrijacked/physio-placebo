# PROGRESS

## Current status
- Semester week: 1–3 sprint (first deliverable = end of Week 3; calendar week of 2026-08-24)
- Phase: foundation — data + harness + classical floor + Week-3 documents
- Last session: 2026-08-27 — repo scaffolded; WESAD end-to-end verified on real data
  (15 subjects, 1,042 windows); features frozen + extracted (zero NaNs); WESAD classical floor
  locked; Health-LLM notes + fidelity-anchor recipe + OSF prereg draft written; SensorLM
  captioning vendored and render-proven on a real WESAD window. All claims below were executed
  and verified this session; commands and outputs are in the session log.

## Week checklist
- [ ] Week 1 — data + harness foundation (**mostly done; blocked on 2 of 3 datasets' raw data**)
  - [x] WESAD downloaded (2.25 GB zip, sha256 recorded), windowed end-to-end: 15/15 subjects,
        1,042 windows (60 s / 30 s stride), 731 non-stress / 311 stress, 0 NaN drops
        (`data/processed/wesad/prepare_manifest.json`)
  - [x] Frozen NeuroKit2 feature extraction (8 HRV + 7 EDA/cvxEDA + 3 RESP): all 1,042 WESAD
        windows, **zero NaNs in all 18 features**
        (`results/features/wesad/20260827T123554Z_84ac4582/`, parquet sha256 `e312262c…`)
  - [x] Feature code frozen: `configs/frozen_features.lock.json`
        (source sha256 `702d5f37…`, neurokit2 0.2.13, freeze-guard test active)
  - [x] Strict LOSO harness + leakage tests + prompt-dev split (`src/physio_placebo/data/loso.py`)
  - [x] MAUS loader written per authors' baseline-repo format; fixture-tested (real data gated —
        see blockers)
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
  - [ ] MAUS raw data: requires IEEE DataPort login → **user action**
- [ ] Week 2 — classical floor (**WESAD done and locked; other datasets pending their data**)
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
  - [ ] MAUS floor (blocked on data)
  - [ ] Byte-identity test (baselines vs LLM feature matrices) — LLM side does not exist yet;
        test lands with the LLM scorer (Week 4)
- [ ] Week 3 — fidelity anchor + OSF pre-registration (**prereg FILED & PUBLIC; anchor run
      pending**)
  - [x] Anchor chosen + recipe + patch list: `docs/fidelity-anchor.md` (PMData stress,
        zero-shot MedAlpaca-7b, target MAE 0.76 ± 0.1) — primary still GPU-gated
  - [ ] **Fallback anchor ATTEMPTED 2026-08-30, blocked at 35/897 calls** by the OpenAI
        account's free-tier cap (50 requests/day/model — see blockers). Everything up to the
        API calls is DONE and verified: PMData extracted, their generator patched + run
        (eval split = 299 items, seed 123, manifest in `results/anchor/split_manifest.json`),
        their inference patched (now checkpoint/resumable), patches recorded in
        `results/anchor/patches.diff`, scorer ready (`scripts/score_anchor.py`). Rerun is
        one command once the account is upgraded (~25 min, ≈$1.30).
  - [x] OSF prereg **FILED & PUBLIC 2026-08-30 18:56 IST**: registration
        https://osf.io/62r5t (DOI 10.17605/OSF.IO/62R5T, Open-Ended Registration,
        OSF Registries; associated project https://osf.io/arj8w). Filed text =
        `docs/osf-prereg.md` @ `f705db9`. Verified publicly visible while logged out,
        2026-08-30 19:10 IST. **The no-LLM-before-prereg gate is CLEARED.**
- [ ] Weeks 4–15: not started (by design — post-prereg)

## Open blockers
- **OpenAI account tier (NEW 2026-08-30, blocks anchor now and the ENTIRE Week 4–10 grid
  later)** — the key's org is on the free tier: **50 requests/day/model** (hit at exactly 50
  during the anchor run; verified via `x-ratelimit` headers: gpt-4.1 and gpt-4.1-mini also
  50/day, with 10k and 60k tokens/min). The anchor needs 897 calls; the grid needs tens of
  thousands. **User action: add a payment method / buy ≥$5 credits at
  platform.openai.com → Settings → Billing** (Tier 1 raises limits to hundreds of
  requests/min). Fallback if never upgraded: checkpointed 50/day trickle ≈ 19 days for the
  anchor alone — not viable for the grid.
- **MAUS raw data** — IEEE DataPort requires a (free) login; user must download
  `MAUS: A Mental Workload Assessment...` (DOI 10.21227/q4td-yd35) and unzip into
  `data/raw/maus/` so `scripts/prepare_maus.py` can run on real data.
- ~~CogWear download~~ **RESOLVED 2026-08-27** via per-file downloads with SHA256 verification
  (`scripts/download_cogwear.py`); the zip resume-loop was abandoned. (This line was stale.)
- ~~PMData~~ **RESOLVED 2026-08-27**: `data/raw/pmdata/pmdata.zip` downloaded complete
  (1,416,129,266 bytes, 912 files, sha256 `53a49d94…` in sidecar). Anchor data side is ready;
  the primary (MedAlpaca) run still needs a GPU box; the fallback anchor
  (few-shot `gpt-3.5-turbo-instruct`, API-only) is runnable on this Mac as of 2026-08-30.
- **Prof sign-off** — verbalizer variant (iv) (SensorLM templates) still pending
  (PROJECT_PLAN.md §11); vendoring done regardless as planned.
- ~~OSF filing~~ **RESOLVED 2026-08-30**: registered and public at https://osf.io/62r5t
  (DOI 10.17605/OSF.IO/62R5T). Filed text = `docs/osf-prereg.md` @ `f705db9`.
- **MAUS access** — IEEE DataPort account-creation page broken for user (2026-08-30); no public
  raw mirror exists (Kaggle copy is processed-PPG only, unusable for our frozen extractor).
  Escalated to prof; fallback = email dataset authors (NTU, addresses in arXiv:2111.02561).
- **GPU hardware** — DOWNGRADED 2026-08-30: the main grid no longer needs a GPU (models swapped
  to OpenAI API snapshots, see decisions log). GPU is now needed only for (a) the MedAlpaca-7b
  primary anchor run and (b) the optional open-weights extension. This Mac (16 GB RAM, 13 GiB
  free disk) still cannot hold MedAlpaca-7b fp16 (~14 GB).

## Decisions log
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
- MAUS on real data (loader is fixture-tested only) — pending user download.
- Fidelity-anchor reproduction run — recipe + patches documented, nothing executed.
- Everything LLM-side (paradigms, prompts, surrogates, API logprob scoring, SDS) —
  intentionally absent until after prereg (Weeks 4+).
- `pip install -e .` required `required_permissions=all` in this sandbox (a `.pth` write);
  fresh-clone install on a second machine still unverified (GPU box now optional).
- Health-LLM releases **no evaluation code** (verified 2026-08-27: no `eval/` dir;
  `medalpaca/` = training/inference utilities only). The paper's MAE must be re-implemented
  for the anchor; convention fixed in `docs/fidelity-anchor.md` §2 and disclosed as part of
  the reproduction gap.
