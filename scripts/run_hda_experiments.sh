#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PYTHON="$ROOT/.venv/bin/python"
CONFIG="$ROOT/code/configs/hda.yaml"
BLUR="$ROOT/artifacts/features_blur_k9.npz"
JPEG="$ROOT/artifacts/features_jpeg_q30.npz"
RESIZE="$ROOT/artifacts/features_resize_25.npz"
RUN_ROOT="$ROOT/runs_hda"
SEEDS=${SEEDS:-"2026 2027 2028"}
cd "$ROOT/code"

for seed in $SEEDS; do
  for weighting in uniform confidence learned; do
    "$PYTHON" -m src.train_hda \
      --blur-features "$BLUR" \
      --jpeg-features "$JPEG" \
      --resize-features "$RESIZE" \
      --weighting "$weighting" \
      --seed "$seed" \
      --output-dir "$RUN_ROOT/hda_${weighting}_seed${seed}" \
      --config "$CONFIG" \
      --device cuda
  done
done

"$PYTHON" -m src.aggregate_hda \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_hda"
