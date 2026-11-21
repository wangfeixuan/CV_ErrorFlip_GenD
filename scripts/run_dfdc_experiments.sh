#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PYTHON="$ROOT/.venv/bin/python"
CONFIG="$ROOT/code/configs/dfdc.yaml"
BLUR="$ROOT/artifacts/dfdc_blur_k9.npz"
JPEG="$ROOT/artifacts/dfdc_jpeg_q30.npz"
RESIZE="$ROOT/artifacts/dfdc_resize_25.npz"
RUN_ROOT="$ROOT/runs_dfdc"
SEEDS=${SEEDS:-"2026 2027 2028"}
cd "$ROOT/code"

evaluate_celebdf() {
  local run_dir="$1"
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external \
    --condition blur_k9 \
    --output "$run_dir/eval_celebdf_blur_k9.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_jpeg_q30.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external \
    --condition jpeg_q30 \
    --output "$run_dir/eval_celebdf_jpeg_q30.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_resize_25.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split external \
    --condition resize_25 \
    --output "$run_dir/eval_celebdf_resize_25.json" \
    --device cuda
}

train_single() {
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
  evaluate_celebdf "$run_dir"
}

for seed in $SEEDS; do
  train_single clean_only "$BLUR" "$seed"
  train_single uniform "$BLUR" "$seed"
  train_single consistency "$BLUR" "$seed"
  train_single restore "$BLUR" "$seed"
  train_single jpeg_only "$JPEG" "$seed"
  train_single resize_only "$RESIZE" "$seed"
  train_single flip_weighted "$BLUR" "$seed"

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
  evaluate_celebdf "$mixed_dir"
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_celebdf" \
  --glob "*/eval_celebdf_*.json"

"$PYTHON" -m src.bootstrap_analysis \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/bootstrap_celebdf.csv" \
  --glob "*/eval_celebdf_*.per_video.csv" \
  --group-manifest "$ROOT/artifacts/features_blur_k9.manifest.csv"
