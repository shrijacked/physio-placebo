#!/usr/bin/env bash
# Zip the minimum tree Colab needs for the MedAlpaca fidelity-anchor run.
# Eval JSON lives under gitignored /data/, so it has to ride in this zip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL="$ROOT/data/third_party/Health-LLM/zero-shot/data/pmdata/stress.json"
if [[ ! -f "$EVAL" ]]; then
  echo "missing locked eval split: $EVAL" >&2
  exit 1
fi
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
DEST_DIR="$STAGING/physio-placebo"
mkdir -p \
  "$DEST_DIR/src/physio_placebo" \
  "$DEST_DIR/scripts" \
  "$DEST_DIR/results/anchor" \
  "$DEST_DIR/data/third_party/Health-LLM/zero-shot/data/pmdata"
cp "$ROOT/pyproject.toml" "$DEST_DIR/"
cp "$ROOT/src/physio_placebo/"*.py "$DEST_DIR/src/physio_placebo/"
cp "$ROOT/scripts/run_medalpaca_anchor.py" "$DEST_DIR/scripts/"
cp "$ROOT/scripts/score_anchor.py" "$DEST_DIR/scripts/"
cp "$ROOT/results/anchor/split_manifest.json" "$DEST_DIR/results/anchor/"
cp "$EVAL" "$DEST_DIR/data/third_party/Health-LLM/zero-shot/data/pmdata/stress.json"
OUT="$ROOT/results/anchor/colab_anchor.zip"
rm -f "$OUT"
(cd "$STAGING" && zip -qr "$OUT" physio-placebo)
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
echo "upload this zip in notebooks/week3_medalpaca_colab.ipynb"
