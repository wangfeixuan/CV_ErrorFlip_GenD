#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
CONFIG="$ROOT/code/configs/wilddeepfake.yaml"
BLUR="$ROOT/artifacts/wilddeepfake_blur_k9.npz"
JPEG="$ROOT/artifacts/wilddeepfake_jpeg_q30.npz"
RESIZE="$ROOT/artifacts/wilddeepfake_resize_25.npz"
RUN_ROOT="$ROOT/runs_wild"
cd "$ROOT/code"

train_adapter() {
  local method="$1"
  local features="$2"
  local seed="$3"
  local run_dir="$RUN_ROOT/${method}_seed${seed}"
  "$PYTHON" -m src.train \
    --features "$features" \
    --method "$method" \
    --seed "$seed" \
    --output-dir "$run_dir" \
    --config "$CONFIG" \
    --device cuda
  evaluate_external "$run_dir" "$seed"
}

evaluate_external() {
  local run_dir="$1"
  local seed="$2"
  local suffix
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external --condition blur_k9 \
    --output "$run_dir/eval_external.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_jpeg_q30.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external --condition jpeg_q30 \
    --output "$run_dir/eval_external_jpeg_q30.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_resize_25.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external --condition resize_25 \
    --output "$run_dir/eval_external_resize_25.json" \
    --device cuda
}

for seed in 2026 2027 2028; do
  train_adapter clean_only "$BLUR" "$seed"
  train_adapter uniform "$BLUR" "$seed"
  train_adapter consistency "$BLUR" "$seed"
  train_adapter restore "$BLUR" "$seed"
  train_adapter jpeg_only "$JPEG" "$seed"
  train_adapter resize_only "$RESIZE" "$seed"

  mixed_dir="$RUN_ROOT/mixed_full_seed${seed}"
  "$PYTHON" -m src.train_mixed \
    --blur-features "$BLUR" \
    --jpeg-features "$JPEG" \
    --resize-features "$RESIZE" \
    --type-strategy all_types \
    --seed "$seed" \
    --output-dir "$mixed_dir" \
    --config "$CONFIG" \
    --device cuda
  evaluate_external "$mixed_dir" "$seed"
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_external"
