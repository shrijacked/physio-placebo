# physio-placebo — Project Plan & Working Log

**Project:** physio-placebo: Is the LLM Reading the Signal, or Is the Verbalizer Doing the Classifying?
**Student:** Shrijak Kumar · **Advisor:** Dr. Siddharth, Plaksha University
**Prof-facing proposal (verbatim, do not edit without approval):** `independent_study_proposal.tex` / `.pdf`
**Provenance / full analysis:** `TOURNAMENT_REPORT.md`
**Semester clock:** Week 1 starts on ______ (fill in once the prof approves) · 15 weeks total
**Doc last updated:** 2026-08-10

---

## 0. How to use this document (instructions for the agent)

1. **Read this file first** in every working session before touching anything else.
2. **Update as you go:** tick checkboxes when a task's acceptance criteria are met (not before), update §10 (Status board) every session, and append to §11 (Decision log) whenever a choice is made that future sessions must respect.
3. **Frozen sections:** after the OSF pre-registration is filed in Week 3, §2 (core question), §5.4 (surrogate ladder definitions), §6 (SDS + validity gate), and §8 (pre-registered predictions) are **FROZEN** — no edits, only dated addenda clearly marked as post-registration.
4. **Never mark work "done" without verification** — a script that runs end-to-end, a number that reproduces, a table that exists. Log the evidence (file path / command / value) next to the checkbox.
5. **The prof-facing PDF wording is fixed.** Any change to the proposal itself needs Shrijak's explicit approval first.
6. **Honesty rule:** if a reproduction gap, a broken surrogate, or a null result shows up, it goes in this doc verbatim. The project is publishable in either direction by design; hiding bad news breaks it.

---

## 1. The idea in one paragraph

A surrogate-controlled audit of the fast-growing "LLMs reason over your physiology" literature: take published prompting pipelines (Health-LLM style), run them under strict leave-one-subject-out (LOSO) on WESAD / MAUS / CogWear, then systematically replace the physiological signal with spectrum-matched noise, a wrong subject's recording, a flatline, or nothing at all, and measure how much of the above-chance margin survives. The sharp part is the second move: separating **verbalizer leakage** (the template says "elevated skin conductance, stress-typical") from actual signal information, which nobody has done.

## 2. The falsifiable core question  *(FROZEN after Week-3 pre-registration)*

> Does an LLM's above-chance performance on physiological state classification survive replacing the signal with spectrum-matched noise, another subject's recording, or nothing at all, and does the feature-verbalized condition keep its advantage once class-suggestive adjectives are stripped from the template — or is the measured performance produced by the prompt scaffolding and the label prior rather than by the physiology?

---

## 3. Datasets (all public, zero gatekeepers)

| Dataset | Source | Subjects | Signals | Task | Notes |
|---|---|---|---|---|---|
| **WESAD** | UCI ML Repository (search "WESAD") | 15 | Chest RespiBAN: ECG/EDA/RESP at 700 Hz; wrist Empatica E4: BVP 64 Hz, EDA 4 Hz | Binary stress vs non-stress (standard task) | The most-walked path in wearable affective computing; classical baselines near ceiling — expect compressed contrasts |
| **MAUS** | IEEE DataPort (free account) | 22 | Wrist PPG (+ ECG, GSR) | Mental workload under n-back | |
| **CogWear** | PhysioNet (search "CogWear") | 11 | Empatica E4 + Muse S | Stroop (cognitive effort) vs baseline | Also carries an EEG channel → stretch modality |
| *Stretch:* DD-Database | Dryad | — | — | — | Only if pipeline is clean by Week 11 |
| *Stretch:* ds003838 | OpenNeuro | — | — | — | Same condition |

**Download discipline:** pull all three Tier-A datasets in Week 1, record checksums and exact versions here. Total volume is a few GB.

## 4. Models & compute

| Role | Model | Config |
|---|---|---|
| Text LLM #1 | Qwen3-8B-Instruct | 4-bit, vLLM, logprob scoring |
| Text LLM #2 | Llama-3.1-8B-Instruct | 4-bit, vLLM, logprob scoring |
| VLM (plot paradigm) | Qwen2.5-VL-7B-Instruct | 4-bit |
| Fine-tuned condition (Week 13) | QLoRA via Unsloth on one 8B model | ~6.6 GB peak at 8B/2048 tokens |
| Paid upper bound (optional) | GPT-class API | Budget < $50 |

**Compute reality check:** every condition is scored by reading **logprobs over the label tokens in a single forward pass**, not free generation. Full grid ≈ 20–40 GPU-hours at 8B/4-bit with vLLM batching. A 16 GB card is sufficient. Verbalizer ablation ≈ 3 days of compute on top.

**Open item:** confirm which GPU we actually have (local card vs university cluster vs rented). → §12 Open questions.

## 5. Experimental design

### 5.1 Paradigms (how the signal reaches the model)
- **Paradigm A — raw numeric series:** downsampled numeric time series pasted into the prompt. *Known risk:* no principled downsampling choice exists (700 Hz ECG / 64 Hz BVP → prompt-sized series is arbitrary); document the chosen scheme, sweep at least two window/rate settings, and report both so "you built it wrong" has an answer.
- **Paradigm B — verbalized features:** NeuroKit2-extracted features rendered into sentences by a template (the condition the literature actually uses). This is where the verbalizer ablation lives.
- **Paradigm C — plot image:** matplotlib rendering of the window → Qwen2.5-VL-7B. WESAD and MAUS only.

### 5.2 Feature extraction (freeze in Week 1)
- NeuroKit2, version-pinned. HRV time + frequency domain; EDA tonic/phasic via cvxEDA; respiration features.
- Same features feed the classical floor and Paradigm B — identical inputs, so the LLM-vs-classical delta is clean.

### 5.3 Classical floor & chance (locked before any LLM runs)
- L2 logistic regression **and** a depth-3 decision tree on the identical features, strict LOSO, per-subject variance reported.
- Chance defined two ways: (a) majority-class, (b) empirical floor from 1000 label permutations.

### 5.4 Surrogate ladder  *(FROZEN after pre-registration)* — applied identically to every paradigm
1. **Gaussian noise**, moments matched to the **globally pooled** distribution (never per-class — per-class matching leaks the label).
2. **IAAFT phase-randomized** surrogate (preserves spectrum/amplitude distribution, destroys phase structure).
3. **Subject-permuted with label alignment broken:** wrong person AND independently sampled condition. *(The naive "wrong person, same condition" version is NOT a null — WESAD's TSST response is stereotyped across people, so it carries real class information.)*
4. **Flatline.**
5. **Signal deleted** — prompt scaffolding only.
6. *Separate, differently-named condition (reported, never counted as a null):* label-preserving subject permutation = "population-level vs individual-level information."

**Information-preservation ledger:** run the classical LR on every surrogate and publish exactly how much class information each surrogate retains. No surrogate's null status is asserted — it is measured.

### 5.5 Prompt protocol (Weeks 4–5, then frozen)
- 3 templates × {0-shot, 4-shot} per condition, selected on a held-out subject fold, then **frozen** before any surrogate is run.
- Each condition gets its own best prompt, so "you prompted it badly" is off the table.

### 5.6 Verbalizer-leakage ablation (Week 9 — the intellectual core)
Template variants on Paradigm B:
- (i) **Original adjectives** ("elevated skin conductance, stress-typical").
- (ii) **Neutralized:** raw values + population percentile, no evaluative words.
- (iii) **Adversarially mis-signed adjectives:** say "low" where the value is high — does the model follow the number or the word?
- (iv) **SensorLM captioning templates** *(added 2026-08-10, pending prof sign-off):* use Google's released hierarchical captioning pipeline (`captioning.py` + `constants.py` from [Google-Health/consumer-health-research/sensorlm](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm), Apache 2.0) as an industry-standard verbalizer variant. This tests verbalizer leakage against the flagship paper's own templates, not just ours. Near-zero extra compute.

**SensorLM status (verified 2026-08-10):** no pretrained weights or API exist for SensorLM (NeurIPS 2025) — repo is reference code only (captioning pipeline + SigLIP training patch for Big Vision); zero checkpoints on Hugging Face; training data (59.7M h Fitbit/Pixel) proprietary. LSM-2 (the newer Google sensor model) is paper-only, no code or weights. **Therefore neither can be a model-under-test; only the captioning templates are usable.**

## 6. Metrics & statistics  *(FROZEN after pre-registration)*

- **Primary metric:** macro-F1 under strict LOSO, with paired-over-subjects bootstrap CIs (1000 resamples, **subject as the resampling unit**) and Wilcoxon signed-rank across subjects. Per-subject variance always reported.
- **SDS (Signal Dependence Score)** = (M_real − M_surrogate) / (M_real − M_chance). **Secondary** ratio only.
- **SDS validity gate:** if a condition's real-signal macro-F1 does not exceed the label-permutation chance floor by a margin whose bootstrap CI excludes zero, SDS is **undefined** for that cell and only the raw delta with CI is reported. No ratio is ever printed on top of a near-zero denominator (the raw-series paradigm is exactly where this bites).
- Classical-baseline delta reported alongside every LLM condition.

## 7. Fidelity anchor (Week 3)

Reproduce **exactly one** Health-LLM headline number using their released code, before any surrogate runs. Report the reproduction gap in the paper regardless of size.
- Week-1 subtask: identify the exact Health-LLM paper, released repo, and which headline number to target. *(Do not guess the citation — verify the repo exists and runs.)*
- Risk note: research-grade reproductions routinely eat 3–4 weeks; the schedule caps this at Week 3 — if it isn't closing, freeze the gap measurement, report it, and move on. Reproduction fidelity is reported, not required.

## 8. Pre-registered predictions (to be falsified)  *(FROZEN at OSF registration)*

1. SDS < 0.3 for raw-series and plot paradigms.
2. The feature-verbalized paradigm loses > 50% of its margin under adjective neutralization.
3. The mis-signed-adjective condition flips predictions in > 40% of items.

OSF pre-registration (Week 3) covers: SDS definition, the surrogate ladder, the SDS-validity gate, and the directional predictions.

---

## 9. 15-week plan with acceptance criteria

> Tick a box only when its **Done means** line is satisfied. Log evidence inline.

### Phase 0 — before Week 1 (now)
- [x] Proposal written, compiled to PDF, shared with prof (2026-08-10)
- [x] SensorLM open-source status investigated; verbalizer-variant idea proposed to prof (2026-08-10)
- [ ] Prof approval to proceed · **Done means:** explicit yes + agreed start date recorded at top of this doc
- [ ] Prof sign-off on verbalizer variant (iv) · **Done means:** yes/no recorded in §11
- [ ] Hardware confirmed · **Done means:** GPU model + VRAM + access path written in §12
- [ ] Repo scaffolded (`physio-placebo/`) with env pinning (Python version, `requirements.txt`/`uv.lock`, NeuroKit2 + cvxEDA + vLLM versions) · **Done means:** fresh-clone install runs on the target machine

### Week 1 — data + harness skeleton
- [ ] Download WESAD, MAUS, CogWear; record versions/checksums in this doc · **Done means:** loader script parses all three, per-subject window counts printed and pasted here
- [ ] NeuroKit2 extraction frozen and version-pinned: HRV time+frequency, EDA tonic/phasic via cvxEDA, respiration · **Done means:** feature matrix per dataset saved; extraction config committed; versions logged
- [ ] LOSO harness built (subject-disjoint folds, deterministic seeds) · **Done means:** unit test proving no subject appears in both train and eval side of any fold
- [ ] Read the Health-LLM released code; pick the headline number for Week 3 · **Done means:** paper + repo URL + chosen number written in §11
- [ ] Locate + vendor SensorLM `captioning.py`/`constants.py`; confirm license terms · **Done means:** templates render on one WESAD window

### Week 2 — classical floor (locked before any LLM runs)
- [ ] L2 logistic regression + depth-3 decision tree, identical features, LOSO, per-subject variance · **Done means:** floor table (3 datasets × 2 classifiers) with per-subject spread, committed
- [ ] Chance both ways: majority-class + 1000-permutation empirical floor · **Done means:** chance table committed; permutation seed logged
- [ ] Sanity: WESAD floor should be strong (tonic EDA mean alone is known to be highly predictive) — if it isn't, the pipeline is broken, stop and debug

### Week 3 — fidelity anchor + pre-registration ⚑ GATE
- [ ] Reproduce one Health-LLM headline number · **Done means:** our number vs published number + gap, in a committed report — gap reported honestly whatever its size
- [ ] OSF pre-registration filed (SDS, surrogate ladder, validity gate, predictions) · **Done means:** OSF URL pasted here; §§2/5.4/6/8 of this doc marked FROZEN
- ⚑ **Gate:** do not run any surrogate condition before the registration timestamp.

### Weeks 4–5 — Paradigms A & B, real signal
- [ ] Prompt-robustness sweep: 3 templates × {0-shot, 4-shot}, selected on held-out subject fold, then frozen · **Done means:** frozen prompt files committed with fold-selection log
- [ ] Paradigm A (raw numeric series) × 2 models × 3 datasets, real signal · **Done means:** macro-F1 + CIs in results table; downsampling scheme(s) documented per §5.1
- [ ] Paradigm B (verbalized features) × 2 models × 3 datasets, real signal · **Done means:** same
- [ ] Logprob label-scoring path verified (no free generation) · **Done means:** per-window label logprobs reproducible across two runs (greedy determinism check)

### Week 6 — Paradigm C
- [ ] Plot rendering pipeline (fixed style, resolution, axes policy) · **Done means:** rendering config committed
- [ ] Paradigm C on WESAD + MAUS with Qwen2.5-VL-7B, real signal · **Done means:** results rows appended

### Weeks 7–8 — surrogate ladder
- [ ] Implement all 5 surrogates + the separate label-preserving condition exactly per §5.4 · **Done means:** each surrogate unit-tested (moments/spectrum/label-alignment assertions)
- [ ] Run ladder identically across all paradigms/models/datasets · **Done means:** full 3×2×3×6 grid populated
- [ ] Information-preservation ledger: classical LR on every surrogate · **Done means:** ledger table committed alongside the grid
- ⚠ Watch: if the label-broken subject permutation still scores high, check the condition resampling — that is a bug signature, not a finding.

### Week 9 — verbalizer-leakage ablation (the core)
- [ ] Variant (i) original adjectives · (ii) neutralized · (iii) adversarially mis-signed · (iv) SensorLM templates (if approved) · **Done means:** ablation panel with CIs for all variants on Paradigm B, all datasets
- [ ] Item-level flip analysis for variant (iii) · **Done means:** flip-rate per dataset committed (feeds prediction #3)

### Week 10 — ★ MINIMUM VIABLE RESULT ⚑ GATE
- [ ] **The SDS report card:** 3 paradigms × 2 open models × 3 datasets × 6 signal conditions, macro-F1 under strict LOSO, paired-over-subjects bootstrap CIs + Wilcoxon, classical-baseline delta, verbalizer-ablation panel · **Done means:** one self-contained document (tables + figures) that would stand as a workshop paper in either result direction
- [ ] Check-in with prof scheduled and held; outcome logged in §11
- ⚑ **Gate:** everything after Week 10 is upside; nothing after this point may jeopardize the MVR artifact.

### Weeks 11–12 — cross-dataset generalization (stretch #1)
- [ ] Train/select verbalizer choices on WESAD, evaluate on MAUS/CogWear · **Done means:** transfer table
- [ ] Stretch datasets (DD-Database, ds003838) only if pipeline is clean · **Done means:** explicit go/no-go note in §11

### Week 13 — QLoRA + API bound (stretch #2, #3)
- [ ] QLoRA-tuned condition via Unsloth: does fine-tuning make the model *start* using the signal (re-run key surrogates on the tuned model)? · **Done means:** tuned-vs-untuned SDS comparison
- [ ] GPT-class API upper bound (< $50) · **Done means:** API rows appended with cost log

### Week 14 — writeup
- [ ] Paper draft. Venue targets: **ACII or ICMI** (affect framing); **ML4H / GenAI4Health workshop** (audit framing) · **Done means:** complete draft with all tables generated from committed scripts (no hand-typed numbers)
- [ ] Limitations section explicitly covers: n=11–22 subjects, WESAD near-ceiling compression, raw-series downsampling arbitrariness, and the §6 validity-gate cells

### Week 15 — release
- [ ] **`physio-placebo` pip package:** wraps any HF model + any physiological dataset into an SDS report card · **Done means:** `pip install` + quickstart runs on a fresh machine against WESAD
- [ ] Reporting checklist (modelled on the EEG-to-text mandatory-noise-baseline recommendation) · **Done means:** checklist shipped in repo + paper appendix
- [ ] Email the reproduction script to the original author group before submission · **Done means:** sent-mail logged in §11

### Stretch backlog (priority order — only after Week-10 MVR is safe)
1. Cross-dataset generalization (Weeks 11–12)
2. QLoRA condition (Week 13)
3. API upper bound (Week 13)
4. EEG channel from CogWear / ds003838 as a fourth modality

---

## 10. Status board  *(update every session)*

**Current week:** pre-semester (Phase 0) · **Overall:** 🟢 on track

| | |
|---|---|
| **Done (latest first)** | 2026-08-10 · Independent Study Registration form filled from this plan + Amol's ref structure (grading adapted, not copied); PDF exported. 2026-08-10 · SensorLM/LSM-2 openness verified (code-only / paper-only; no weights) — reply sent to prof; verbalizer variant (iv) proposed. 2026-08-10 · Proposal PDF finalized (colored, named) and shared with prof. 2026-07-28 · Idea selection completed; winner + runner-up fixed. |
| **In progress** | Awaiting student ID No / course-no confirmation for the registration form; awaiting prof approval + start date. |
| **Blocked** | Registration form PDF ready except ID No (blank until confirmed). |
| **Next 3 actions** | 1) Confirm ID No + Course No (CS4002 vs CS4003) + Major/Category → regenerate form PDF. 2) Get prof approval + start date → fill in semester clock. 3) Confirm GPU hardware (§12). |

## 11. Decision log  *(append-only; date every entry)*

| Date | Decision | Rationale / evidence |
|---|---|---|
| 2026-07-28 | Project = physio-placebo (surrogate-controlled audit); runner-up = gaze-grounded reward-model audit (see proposal §7) | Idea-selection analysis in `TOURNAMENT_REPORT.md` |
| 2026-08-10 | Proposal wording locked; prof-facing PDF contains no selection-process language | Shrijak's instruction |
| 2026-08-10 | SensorLM cannot be a model-under-test (no public weights for SensorLM or LSM-2; verified via official repo + Hugging Face API) | Repo contains only `captioning.py`, `constants.py`, SigLIP training patch; HF search returns 0 models |
| 2026-08-10 | Proposed ablation variant (iv): SensorLM captioning templates as industry-standard verbalizer — **pending prof sign-off** | Strengthens verbalizer-leakage claim at near-zero compute; Apache 2.0 license permits reuse |
| 2026-08-10 | Registration form grading adapted from Amol's ref structure to experimental deliverables: Reproduction+Pre-reg 20% / Mid-semester SDS report 30% / Final paper 30% / Open-source release 10% / Weekly engagement 10% | Survey-style bibliography/critical-analysis split from the ref does not fit this audit project; weights still sum to 100% with a Week-3 + Week-10 gate structure matching §9 |
| | *(next entries go here)* | |

## 12. Open questions  *(resolve and move answers to §11)*

1. **Start date / semester alignment** — which week is Week 1?
2. **Hardware** — exact GPU (need ≥16 GB VRAM); local vs cluster; vLLM compatibility.
3. **Health-LLM anchor** — exact paper, repo URL, and which headline number we reproduce (decide in Week 1 after reading the released code; do not cite from memory).
4. **Prof sign-off** on verbalizer variant (iv) and on the Week-10 check-in format.
5. **OSF account** — create under Shrijak's name; decide whether prof is listed as collaborator on the registration.
6. **Raw-series downsampling scheme** for Paradigm A — pick + document two candidate schemes in Week 4 (this is the reviewer's main attack surface on the null).

## 13. Risk register (from proposal §5, operationalized)

| Risk | Symptom | Mitigation (built into design) |
|---|---|---|
| Surrogates aren't nulls | High performance on a "null" condition | Globally-pooled moment matching; label-broken subject permutation; label-preserving version reported separately; information-preservation ledger measures every surrogate |
| "You built it wrong" dismissal of a null | Reviewer blames prompting or SDS instability | Week-3 fidelity anchor vs released code; per-condition best prompt, frozen pre-surrogate; SDS validity gate (no ratio over near-zero denominator); subject-level resampling everywhere |
| "Everyone already knew this" | True-but-boring rejection | The published literature claims otherwise and reports gains; the verbalizer arm is non-obvious in either direction; pre-registration timestamps the predictions |
| Fidelity anchor won't reproduce | Weeks slip in Week 3 | Hard cap: report the gap at Week 3 and proceed — the gap is a datum, not a blocker |
| WESAD near-ceiling squeeze | Compressed LLM-vs-floor contrasts | Three datasets, not one; report absolute macro-F1 as primary, SDS secondary |
| Small-n instability (11–22 subjects) | Overlapping CIs everywhere | Subject-level bootstrap honestly reflects it; Wilcoxon across subjects; per-subject spread always shown |

## 14. Reference shelf

- **Jo et al., "Are EEG-to-Text Models Working?"** — arXiv 2405.06459. The audit discipline this project imports (noise-baseline ablation exposing teacher forcing).
- **Health-LLM line** — prompting pipelines that verbalize wearable features; report gains, no signal ablation. *(Pin exact citation + repo in Week 1.)*
- **Tan et al., "Are Language Models Actually Useful for Time Series Forecasting?"** — NeurIPS 2024. Genre precedent: ablate-the-LLM audits are publishable at strong venues.
- **SensorLM** — arXiv 2506.09108, NeurIPS 2025. Sensor–language foundation models; code-only release (captioning pipeline + SigLIP patch, Apache 2.0): [repo](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm). Benchmarks against other models, never against surrogate signals.
- **LSM-2 (AIM)** — arXiv 2506.05321. Google's newer sensor foundation model; paper-only, no code or weights.
- **VitalAgent** — arXiv 2605.29483. Closest 2026 work: leakage-free vs oracle context comparison — a capability upper bound, not a surrogate control.
- **Datasets:** WESAD (UCI), MAUS (IEEE DataPort), CogWear (PhysioNet); stretch: DD-Database (Dryad), OpenNeuro ds003838.
- **Fallback idea** (only if this project dies at a gate): gaze-grounded reward-model audit — proposal §7.

---

## 15. Check-in template (paste + fill for each prof meeting)

```markdown
### Check-in — <date> (Week <n>)
**Since last time:** <2–4 bullets, each with evidence links>
**Headline numbers:** <table or "none yet">
**On/off track vs §9:** <on track / slipping — where and why>
**Decisions needed from prof:** <bullets>
**Next until following check-in:** <bullets mapped to §9 checkboxes>
**Risks that moved:** <ref §13 rows, or "none">
```
