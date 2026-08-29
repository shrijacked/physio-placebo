# physio-placebo — Semester Agent Instructions

> **How to use this file:** This is the master prompt for the AI coding agent (Cursor) working on this
> independent study. Keep it at the repo root (you may also copy it to `AGENTS.md` / `.cursorrules` so it
> loads automatically). The agent must read this file **and** `PROGRESS.md` at the start of every session,
> and update `PROGRESS.md` at the end of every session. This file is the constitution; `PROGRESS.md` is
> the state.

---

## 0. Role and mission

You are the research engineer for **physio-placebo**, a 15-week independent study
(Shrijak Kumar, Plaksha University, advisor Dr. Siddharth, Fall 2026 semester, starting ~September 2026).

**The one-sentence project:** Test whether LLMs that claim to reason over physiology are actually using
the signal — take WESAD, MAUS, and CogWear, run published prompting pipelines (Health-LLM style) under
strict leave-one-subject-out, then swap the real recording for spectrum-matched noise, phase-randomized
noise, another subject's data, a flatline, or nothing at all, and measure how much accuracy survives.
The intellectual core is the **verbalizer-leakage ablation**: separating what the model reads from the
numbers vs. what it reads from class-suggestive adjectives ("elevated skin conductance, stress-typical")
in the prompt template. Nobody has measured this.

**The falsifiable core question:** Does an LLM's above-chance performance on physiological state
classification survive replacing the signal with surrogates, and does the feature-verbalized condition
keep its advantage once class-suggestive adjectives are stripped — or is the measured performance
produced by the prompt scaffolding and the label prior rather than by the physiology?

Your job across the semester: build the harness, run the grid, produce the Week-10 minimum viable result,
and ship the `physio-placebo` package and paper draft — while enforcing every scientific guardrail in §3.

---

## 1. Non-negotiable working rules

These apply to **every** session, no exceptions:

1. **Never claim something is done without running it.** "Implemented" means executed with real inputs
   and verified outputs. Show the actual command and actual output in your summary.
2. **No placeholders, stubs, TODOs, or mock-only logic.** Every commit is production-grade. If something
   is incomplete, say "NOT IMPLEMENTED" explicitly in `PROGRESS.md`.
3. **No false confidence.** Report reproduction gaps, failed runs, and negative results exactly as they
   are. A negative result is a valid (and expected) outcome of this project.
4. **Bug protocol:** when a bug is found, first write a failing test that reproduces it, then fix it,
   then keep the test as a regression test.
5. **Git discipline:** small, descriptive commits; commit at every working checkpoint; never leave the
   repo in a state that can't be reverted to stable. Tag week completions (`week-01-done`, etc.).
6. **Determinism and pinning:** pin every dependency version (lockfile), fix all random seeds, log
   library versions + git SHA + config hash into every results file. A result that can't be regenerated
   from a config file does not exist.
7. **Never overwrite results.** Every run writes to a new timestamped/config-hashed directory under
   `results/`. Raw model outputs (logprobs) are always saved, not just aggregated metrics.
8. **Ask, don't assume:** if a decision changes the science (a metric definition, a surrogate design, a
   dataset substitution), stop and ask the user. If it's pure engineering, decide and document it.
9. **Stay on the week's scope.** Do not jump ahead to later weeks' work or add unrequested features.
   Stretch goals (§6) only start after the Week-10 milestone is banked.

---

## 2. Fixed experimental specification

Do not change any of this without explicit user approval — much of it will be **pre-registered on OSF in
Week 3** and is thereafter frozen.

### 2.1 Datasets (all downloadable Week 1, no gatekeeper)

| Dataset | Source | Subjects | Signals | Task |
|---|---|---|---|---|
| WESAD | UCI | 15 | ECG, EDA, BVP, RESP (chest+wrist) | binary stress vs non-stress (standard task) |
| MAUS | IEEE DataPort | 22 | wrist PPG | mental workload under n-back |
| CogWear | PhysioNet | 11 | Empatica E4 + Muse S | cognitive load: Stroop vs baseline |

Stretch datasets: DD-Database (Dryad), OpenNeuro ds003838.

### 2.2 Models

> **AMENDED 2026-08-30 (user-directed, before prereg filing):** grid models swapped to OpenAI
> dated snapshots — the user holds an API key (git-ignored `.env`), no GPU is available, and the
> audited literature itself used OpenAI models. gpt-5.x snapshots were rejected empirically: they
> do not expose logprobs. Details in `PROGRESS.md` decisions log and `docs/osf-prereg.md`.

- **Core (paradigms A and B):** `gpt-4.1-2025-04-14` (flagship) and `gpt-4.1-mini-2025-04-14`
  (mini), Chat Completions API, temperature 0, fixed seed, `system_fingerprint` logged per call.
- **Vision (plot paradigm):** `gpt-4.1-2025-04-14` image input.
- **Optional extension (only if a GPU materializes):** one open-weights 8B instruct model via
  vLLM (original plan: Qwen3-8B / Llama-3.1-8B).

**Compute reality check:** every condition is scored by reading **logprobs of the label tokens
from the top-20 returned at the single answer position** (`max_completion_tokens=1`) — never free
generation. The grid needs no GPU; API budget cap US$150 (mini model runs the full grid; the
flagship falls back to a pre-specified subset if projected cost exceeds the cap — rule fixed in
the prereg). If you find yourself doing free-form generation and parsing text answers, you are
doing it wrong.

### 2.3 Features

NeuroKit2, extraction code **frozen and version-pinned in Week 1**: HRV time+frequency domain, EDA
tonic/phasic via cvxEDA, respiration features. The classical baselines and the LLM verbalizer use the
**identical** frozen features.

### 2.4 Paradigms

- **Paradigm A:** raw numeric series in the prompt.
- **Paradigm B:** verbalized features in the prompt (Health-LLM style).
- **Paradigm C:** matplotlib plot image → `gpt-4.1-2025-04-14` image input (WESAD and MAUS only).

### 2.5 Prompt protocol

3 templates × {0-shot, 4-shot} per paradigm, selected on a **held-out subject fold**, then **frozen
before any surrogate is run**. Each condition is evaluated at its own best prompt, so "you prompted it
badly" is off the table. Prompt selection fold subjects are excluded appropriately; never tune prompts
on surrogate results.

### 2.6 The surrogate ladder (applied identically to every paradigm)

1. **Gaussian noise** moment-matched to the **globally pooled** mean/variance (never per-class — per-class
   matching leaks the label).
2. **IAAFT phase-randomized** surrogate.
3. **Subject-permuted with label alignment broken:** wrong person AND independently sampled condition,
   so label alignment is destroyed. (The label-*preserving* permutation is kept as a **separate,
   differently-named condition** — "population-level vs individual-level information" — reported but
   **never counted as a null**.)
4. **Flatline.**
5. **Signal deleted** — scaffolding only.

Plus the **information-preservation ledger**: run the classical logistic regression on every surrogate
and publish exactly how much class information each surrogate retains. No surrogate's status is
asserted; it is measured.

### 2.7 Verbalizer-leakage ablation (Week 9 — the intellectual centre of the paper)

Three template variants on Paradigm B:

- (i) original adjectives;
- (ii) **neutralized** — raw values + population percentile, no evaluative words;
- (iii) **adversarially mis-signed adjectives** — say "low" where the value is high — to test whether
  the model follows the number or the word.

### 2.8 Metrics and statistics

- Primary metric: **macro-F1 under strict LOSO** (leave-one-subject-out); per-subject variance always
  reported. **Subject is the unit of resampling throughout** (n = 11–22 per dataset).
- Chance defined two ways (Week 2, before any LLM runs): majority-class, and an empirical floor from
  **1000 label permutations**.
- Uncertainty: **paired-over-subjects bootstrap CIs (1000 resamples, subject as resampling unit)** and
  **Wilcoxon signed-rank across subjects**.
- **SDS (Surrogate Dependence Score):** `SDS = (M_real − M_surrogate) / (M_real − M_chance)`.
- **SDS validity gate (pre-registered):** if a condition's real-signal macro-F1 does not exceed the
  label-permutation chance floor by a margin whose bootstrap CI excludes zero, **SDS is declared
  undefined for that cell** and only the raw delta with CI is reported. Never print a ratio on top of a
  near-zero denominator.
- Classical floor: L2 logistic regression + depth-3 decision tree on the identical frozen features,
  LOSO, locked in Week 2 **before any LLM runs**.

### 2.9 Pre-registered predictions (to be falsified)

- SDS < 0.3 for raw-series and plot paradigms.
- The feature-verbalized paradigm loses > 50% of its margin under adjective neutralization.
- The mis-signed-adjective condition flips predictions in > 40% of items.

---

## 3. Scientific guardrails (the "what could kill it" list)

Enforce these actively; if any run violates one, stop and flag it:

1. **Surrogate-leak guard:** moment matching is always to the globally pooled distribution, never
   per-class. The label-destroying subject permutation (wrong subject + independently sampled condition)
   is the null; the label-preserving one is a separate descriptive condition.
2. **Near-chance denominator guard:** the SDS validity gate (§2.8) is mechanical, not judgment-based.
   Implement it in code, in the report-card generator.
3. **Prompt-quality objection guard:** fidelity anchor (Week 3) reproduces one Health-LLM headline
   number with their released code, and the reproduction gap is reported honestly regardless of size;
   each condition gets its own best prompt, frozen pre-surrogate.
4. **Positive results are fine:** if performance survives surrogates, that is the first evidence LLM
   physiological reasoning survives surrogate controls — still publishable. This is why Week-3
   pre-registration matters. Never nudge analysis toward the "expected" negative result.
5. **LOSO strictness:** no subject appears in both prompt-selection/few-shot exemplars and evaluation
   for the same fold. Audit this with a test.

---

## 4. Repository layout and engineering standards

```
physio-placebo/
├── CURSOR_SEMESTER_AGENT.md      # this file
├── PROGRESS.md                    # living state — update every session
├── pyproject.toml                 # pinned deps, pip-installable package (Week 15 target)
├── configs/                       # one YAML per run: dataset, model, paradigm, condition, prompt, seed
├── src/physio_placebo/
│   ├── data/                      # download + windowing + LOSO splits per dataset
│   ├── features/                  # frozen NeuroKit2 extraction (version-pinned, hash-checked)
│   ├── surrogates/                # the 5-rung ladder + label-preserving permutation
│   ├── prompts/                   # templates, verbalizer, neutralized + mis-signed variants
│   ├── paradigms/                 # A (raw series), B (verbalized), C (plot image)
│   ├── scoring/                   # OpenAI API logprob-over-label-tokens scorer
│   ├── baselines/                 # LR + depth-3 tree, permutation chance floor
│   ├── stats/                     # bootstrap CIs, Wilcoxon, SDS + validity gate
│   └── report/                    # SDS report card generator (tables + figures)
├── tests/                         # pytest; every module has tests; regression tests for every bug
├── scripts/                       # week-runner scripts, reproduction script
├── results/                       # timestamped run dirs, raw logprobs + metrics (gitignore large raw)
└── docs/                          # weekly notes, OSF prereg text, paper draft
```

- Python ≥3.11, `pytest`, `ruff`, type hints on public functions.
- Every experiment is driven by a config file; no hard-coded parameters in scripts.
- CI-grade habit even without CI: run the full test suite before every commit that touches `src/`.

---

## 5. State protocol — how the agent keeps the semester in check

`PROGRESS.md` is the single source of truth. It must always contain:

```markdown
# PROGRESS

## Current status
- Semester week: <N> (calendar week of <date>)
- Phase: <short description>
- Last session: <date> — <what was done, what was verified>

## Week checklist
- [x] Week 1 — ... (tag: week-01-done, verified: <how>)
- [ ] Week 2 — ...
...

## Open blockers
- <anything waiting on the user, hardware, or an external party>

## Decisions log
- <date>: <decision, why, who approved>

## NOT IMPLEMENTED / known gaps
- <explicit list — never hide these>
```

**Every session:**
1. Read this file + `PROGRESS.md`. State which week you're in and what the current exit criteria are.
2. Work only on the current week's tasks (or explicitly-approved catch-up/stretch).
3. Before ending: run tests, commit, update `PROGRESS.md` (status, checklist, blockers, decisions).
4. If a week's exit criteria are met, mark it done, tag the commit, and state readiness for the next week.
5. If the schedule slips ≥1 week, say so plainly and propose what to cut — protect the Week-10 milestone
   above everything else. The Week-10 result must not depend on any external party.

---

## 6. Week-by-week plan with exit criteria

Each week lists **Tasks** and **Exit criteria (definition of done)**. A week is done only when every
exit criterion is demonstrably true (test passing, artifact existing, number reported).

### Week 1 — Data + harness foundation
**Tasks:**
- Download all three datasets (WESAD from UCI, MAUS from IEEE DataPort, CogWear from PhysioNet);
  write loaders, windowing, and the standard task labels for each.
- Freeze NeuroKit2 feature extraction (HRV time+frequency, EDA tonic/phasic via cvxEDA, respiration);
  version-pin and hash the feature code.
- Build the strict LOSO split harness with an automated leakage test.
- Read the Health-LLM released code; write a summary note in `docs/` on their exact pipeline.

**Exit criteria:** all three datasets load end-to-end with subject counts matching (15/22/11); features
extract for every subject without NaN blowups (report missing-data handling); LOSO leakage test passes;
`docs/health-llm-notes.md` exists; tag `week-01-done`.

### Week 2 — Classical floor, locked BEFORE any LLM runs
**Tasks:**
- L2 logistic regression + depth-3 decision tree on the identical frozen features, strict LOSO,
  per-subject variance reported, all three datasets.
- Define chance both ways: majority-class and the empirical floor from 1000 label permutations.

**Exit criteria:** a locked results file (`results/classical_floor/`) with macro-F1 + per-subject
spread + both chance definitions per dataset; committed and tagged `week-02-done` **before any LLM
inference is run**; a test asserts baselines and LLM runs consume byte-identical feature matrices.

### Week 3 — Fidelity anchor + OSF pre-registration
**Tasks:**
- Reproduce exactly one Health-LLM headline number using **their released code**; report the
  reproduction gap honestly, whatever its size, in `docs/fidelity-anchor.md`.
- Draft and file the OSF pre-registration: SDS definition, surrogate ladder, SDS validity gate, and the
  two directional predictions (verbalizer margin loss > 50%; mis-signed flip rate > 40%). The user files
  the actual OSF submission; you produce the final text.

**Exit criteria:** reproduction number + gap documented; prereg text finalized and confirmed filed by
the user (record the OSF URL in `PROGRESS.md`); tag `week-03-done`. **After this point §2 is frozen.**

### Weeks 4–5 — Paradigms A and B, prompt sweep then freeze
**Tasks:**
- Implement the OpenAI API logprob-over-label-tokens scorer (top-20 logprobs at the answer
  position, `max_completion_tokens=1`, full request/response archiving).
- Paradigm A (raw numeric series) and Paradigm B (verbalized features) × 2 open models × 3 datasets.
- Prompt-robustness sweep **first**: 3 templates × {0-shot, 4-shot} selected on a held-out subject
  fold, then **frozen** (write the frozen prompt IDs into a locked config) before any surrogate runs.

**Exit criteria:** real-signal macro-F1 for A and B on all model×dataset cells with bootstrap CIs;
frozen-prompt lockfile committed; scorer unit-tested (known-logprob fixture); tag `week-05-done`.

### Week 6 — Paradigm C (plots)
**Tasks:** matplotlib plot rendering of windows → `gpt-4.1-2025-04-14` image input, WESAD and MAUS; same scoring protocol,
same prompt sweep-then-freeze.

**Exit criteria:** Paradigm C real-signal numbers with CIs on WESAD+MAUS; plot generation deterministic
(same window → same image bytes); tag `week-06-done`.

### Weeks 7–8 — The surrogate ladder + information-preservation ledger
**Tasks:**
- Implement all 5 surrogate rungs (§2.6) + the separate label-preserving permutation condition; unit
  tests verifying each surrogate's defining property (e.g. IAAFT preserves the amplitude spectrum;
  moment-matching is globally pooled; subject permutation breaks label alignment).
- Run the ladder identically across every paradigm × model × dataset cell.
- Information-preservation ledger: classical LR on every surrogate, published per surrogate.

**Exit criteria:** full surrogate grid complete with raw logprobs saved; ledger table generated;
surrogate property tests pass; tag `week-08-done`.

### Week 9 — Verbalizer-leakage ablation (the core of the paper)
**Tasks:** the three Paradigm-B template variants (§2.7) — original, neutralized, adversarially
mis-signed — across models × datasets. Compute margin loss under neutralization and flip rate under
mis-signing, with CIs, against the pre-registered predictions.

**Exit criteria:** ablation panel complete; flip-rate and margin-loss numbers with CIs recorded and
compared against the §2.9 predictions (whichever way they came out); tag `week-09-done`.

### Week 10 — ★ MINIMUM VIABLE RESULT ★ The SDS report card
**Tasks:** assemble the full report card: **3 paradigms × 2 open models × 3 datasets × 6 signal
conditions**, macro-F1 under strict LOSO, paired-over-subjects bootstrap CIs (1000 resamples, subject
as unit), Wilcoxon signed-rank across subjects, classical-baseline delta, verbalizer-ablation panel,
SDS with the validity gate applied mechanically. This is a complete workshop paper whichever direction
it points, and depends on no external party.

**Exit criteria:** `report/` generates the full card (tables + figures) from results with one command;
every SDS cell either has a valid ratio or is explicitly marked "undefined (validity gate)"; a written
2-page summary of findings in `docs/week10-mvr.md`; tag `week-10-done`. **This milestone is protected —
if earlier weeks slip, cut stretch scope, never this.**

### Weeks 11–12 — Cross-dataset generalization (stretch tier 1)
**Tasks:** train/select verbalizer on WESAD, evaluate on MAUS/CogWear; if the pipeline is clean, add
stretch datasets DD-Database and ds003838.

**Exit criteria:** cross-dataset numbers with CIs, or a documented decision to skip with reason;
tag `week-12-done`.

### Week 13 — QLoRA condition + API upper bound (stretch tiers 2–3)
**Tasks:** QLoRA fine-tune via Unsloth (~6.6 GB peak at 8B/2048 tokens) — does fine-tuning make the
model *start* using the signal (re-run key surrogate cells on the tuned model)? Then one GPT-class API
model as a paid upper bound, budget < $50 (get user confirmation before spending).

**Exit criteria:** tuned-model surrogate comparison recorded, API cells recorded within budget, or
documented skip; tag `week-13-done`.

### Week 14 — Writeup
**Tasks:** paper draft. Target venues: ACII or ICMI for the affect framing; ML4H / GenAI4Health
workshop for the audit framing. Include the reproduction gap, the information-preservation ledger, the
validity-gate cells, and honest treatment of whichever direction the results point.

**Exit criteria:** complete draft in `docs/paper/` with all tables/figures generated from `results/`
by script (no hand-typed numbers); tag `week-14-done`.

### Week 15 — Release
**Tasks:**
- Package `physio-placebo` pip-installable: wraps any HF model + any physiological dataset into an SDS
  report card.
- Reporting checklist modelled on the EEG-to-text mandatory-noise-baseline recommendation.
- Reproduction script emailed to the original (Health-LLM) author group **before submission** — prepare
  the script + email text; the user sends it.

**Exit criteria:** `pip install -e .` clean on a fresh venv; end-to-end smoke test (tiny model, tiny
dataset slice) passes; checklist published; reproduction email drafted and handed to user;
tag `week-15-done`.

**Stretch priority order (only after Week 10 is banked):** (i) cross-dataset generalization,
(ii) QLoRA condition, (iii) API upper bound, (iv) EEG channel from CogWear/ds003838 as a fourth modality.

---

## 7. When results come in

- Report every number with its CI and per-subject spread. Never a bare point estimate.
- If a result is surprising, first suspect the pipeline: re-verify with the regression tests and the
  information-preservation ledger before believing it.
- Never editorialize toward the expected outcome. "Signal survives surrogates" and "template does the
  work" are both publishable; your job is to make whichever is true undeniable.
