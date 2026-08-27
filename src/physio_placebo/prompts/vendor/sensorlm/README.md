# Vendored: SensorLM captioning pipeline

Vendored verbatim (no modifications) for verbalizer ablation variant (iv) — the
"industry-standard verbalizer" condition (PROJECT_PLAN.md §5, pending prof sign-off for use;
vendoring itself is the Week-1 task).

- Upstream: <https://github.com/Google-Health/consumer-health-research> (`sensorlm/`)
- Commit: `7d087e9b93bcd09ce093a5004de8554588fb0522` (fetched 2026-08-27)
- Files: `captioning.py`, `constants.py` (both unmodified), `UPSTREAM_README.md` (upstream
  README, which contains the license grant)
- License: Apache License 2.0, per the License section of `UPSTREAM_README.md`. No LICENSE file
  exists at the upstream repo root; the README section is the operative grant.
- Upstream has **no pretrained weights or API** (verified 2026-08-10, PROJECT_PLAN.md §5) —
  only these captioning templates are usable; SensorLM cannot be a model-under-test.

Import as `physio_placebo.prompts.vendor.sensorlm.captioning`. The
`NORMALIZATION_PARAMETERS` / `FEATURES_TO_NORMALIZE` constants define Google's Fitbit feature
space; our adapter (Week 9) maps the frozen physio-placebo features into the overlapping
channels (HR, rr/sdnn/rmssd/pnn, LF/HF, EDA level).
