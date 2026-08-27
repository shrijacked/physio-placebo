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
- [ ] Week 3 — fidelity anchor + OSF pre-registration (**documents drafted; runs pending**)
  - [x] Anchor chosen + recipe + patch list: `docs/fidelity-anchor.md` (PMData stress,
        zero-shot MedAlpaca-7b, target MAE 0.76 ± 0.1) — **reproduction NOT RUN yet**
  - [x] OSF prereg full draft: `docs/osf-prereg.md` (placeholders: OSF URL, repo URL,
        anchor-run disclosure) — **not filed; user files it**
- [ ] Weeks 4–15: not started (by design — post-prereg)

## Open blockers
- **MAUS raw data** — IEEE DataPort requires a (free) login; user must download
  `MAUS: A Mental Workload Assessment...` (DOI 10.21227/q4td-yd35) and unzip into
  `data/raw/maus/` so `scripts/prepare_maus.py` can run on real data.
- **CogWear download** — PhysioNet static zip (187,856,113 bytes) keeps resetting mid-stream;
  a 40-attempt resume loop is running. If it dies, rerun the same command (it resumes).
- ~~PMData~~ **RESOLVED 2026-08-27**: `data/raw/pmdata/pmdata.zip` downloaded complete
  (1,416,129,266 bytes, 912 files, sha256 `53a49d94…` in sidecar). Anchor data side is ready;
  only the GPU box remains for the anchor run.
- **Prof sign-off** — verbalizer variant (iv) (SensorLM templates) still pending
  (PROJECT_PLAN.md §11); vendoring done regardless as planned.
- **OSF filing** — user reviews `docs/osf-prereg.md`, resolves ⟨placeholders⟩, files, and
  records the OSF URL here.
- **GPU hardware** — still unconfirmed (PROJECT_PLAN.md §12); needed from Week 4 (vLLM) and
  **required for the anchor run**: this Mac has 16 GB RAM and 13 GiB free disk, which cannot
  hold MedAlpaca-7b fp16 (~14 GB). Verified 2026-08-27.

## Decisions log
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
- CogWear loader/windowing/floor — pending the download (loader will be written against the
  real zip, not guessed).
- MAUS on real data (loader is fixture-tested only) — pending user download.
- Fidelity-anchor reproduction run — recipe + patches documented, nothing executed.
- Everything LLM-side (paradigms, prompts, surrogates, vLLM scoring, SDS) — intentionally
  absent until after prereg (Weeks 4+).
- `pip install -e .` required `required_permissions=all` in this sandbox (a `.pth` write);
  fresh-clone install on the target GPU machine still unverified.
- Health-LLM releases **no evaluation code** (verified 2026-08-27: no `eval/` dir;
  `medalpaca/` = training/inference utilities only). The paper's MAE must be re-implemented
  for the anchor; convention fixed in `docs/fidelity-anchor.md` §2 and disclosed as part of
  the reproduction gap.
