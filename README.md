# physio-placebo

Surrogate-controlled audit of LLM physiological-state classification on wearable data
(WESAD, MAUS, CogWear). Independent study, Plaksha University.

The question: does an LLM's above-chance performance survive replacing the physiological
signal with spectrum-matched noise, another subject's recording, or nothing at all — or is
the measured performance produced by prompt scaffolding and the label prior?

## Status

**Pre-registered:** [osf.io/62r5t](https://osf.io/62r5t)
(DOI [10.17605/OSF.IO/62R5T](https://doi.org/10.17605/OSF.IO/62R5T), filed 2026-08-30).

WESAD and CogWear are windowed, features frozen, and classical floors locked. MAUS is blocked
on an IEEE DataPort login. The LLM grid (pinned OpenAI `gpt-4.1` snapshots, logprob scoring)
starts Week 4 — the pre-registration gate is cleared.

Locked numbers and blockers live in [`PROGRESS.md`](PROGRESS.md). The design is in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Setup

Python 3.11+. From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Raw datasets are **not** in git (they are several GB). Download them locally:

| Dataset | How |
|---|---|
| WESAD | UCI ML Repository zip → `data/raw/wesad/WESAD.zip` |
| CogWear | `python scripts/download_cogwear.py` (PhysioNet, SHA256-verified) |
| MAUS | IEEE DataPort (free login) → `data/raw/maus/` |
| PMData (fidelity anchor) | public zip → `data/raw/pmdata/pmdata.zip` |

Then:

```bash
python scripts/prepare_wesad.py
python scripts/prepare_cogwear.py
python scripts/extract_features.py --dataset wesad
python scripts/run_classical_floor.py --dataset wesad
```

Windowing and floor configs are in `configs/`. Feature code is frozen by
`configs/frozen_features.lock.json`; changing `src/physio_placebo/features/frozen.py`
without re-freezing fails `tests/test_feature_hash.py`.

## Layout

```
src/physio_placebo/   loaders, LOSO, frozen features, classical floor, provenance
scripts/              download / prepare / extract / floor runners
tests/                fixture-based; does not need the real GB-scale archives
results/              committed metrics, manifests, parquet feature tables
docs/                 Health-LLM notes, fidelity-anchor recipe, OSF prereg (filed 2026-08-30)
```

`results/**/*.npz` stay on disk only. Everything else under `results/` that is a metric
or manifest is committed so numbers are reproducible from git.
