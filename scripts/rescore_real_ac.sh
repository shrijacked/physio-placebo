#!/usr/bin/env bash
# Re-score real-signal Paradigm A and C with per-minute logprobs saved.
# Run on the A6000, from a git checkout, so the results record the commit.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "git $(git rev-parse HEAD)"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/eval_grid_rows/${STAMP}"
mkdir -p "$OUT"
PY=python
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
fi
# vLLM 0.29 wants 0.92 of the card. Other jobs on this A6000 leave about
# 41 GiB free of 47, so 0.92 does not start. 0.80 still fits an 8B fp16 model.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
"$PY" scripts/run_eval_grid.py --client vllm --model qwen3_8b --paradigm A --save-rows --out-dir "$OUT"
"$PY" scripts/run_eval_grid.py --client vllm --model llama31_8b --paradigm A --save-rows --out-dir "$OUT"
"$PY" scripts/run_eval_grid.py --client vllm --model qwen25_vl --paradigm C --save-rows --out-dir "$OUT"
echo "wrote $OUT"
