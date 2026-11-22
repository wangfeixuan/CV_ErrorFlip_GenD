#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

train_and_evaluate() {
  local name="$1"
  local method="$2"
  local feature_file="$3"
  local seed="$4"
  local run_dir="$ROOT/runs_final/${name}_seed${seed}"

  "$PYTHON" -m src.train \
    --features "$feature_file" \
    --method "$method" \
    --seed "$seed" \
    --output-dir "$run_dir" \
    --device cuda

  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split test --condition blur_k9 \
    --output "$run_dir/eval_test.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_jpeg_q30.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split test --condition jpeg_q30 \
    --output "$run_dir/eval_test_jpeg_q30.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_resize_25.npz" \
    --checkpoint "$run_dir/best.pt" \
    --split test --condition resize_25 \
    --output "$run_dir/eval_test_resize_25.json" \
    --device cuda
}

for seed in 2026 2027 2028; do
  train_and_evaluate \
    jpeg_only \
    jpeg_only \
    "$ROOT/artifacts/features_jpeg_q30.npz" \
    "$seed"
  train_and_evaluate \
    resize_only \
    resize_only \
    "$ROOT/artifacts/features_resize_25.npz" \
    "$seed"
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$ROOT/runs_final" \
  --output-prefix "$ROOT/runs_final/aggregate_all_conditions"
"$PYTHON" -m src.bootstrap_analysis \
  --runs-dir "$ROOT/runs_final" \
  --output "$ROOT/runs_final/bootstrap_all_conditions.csv"
"$PYTHON" -m src.make_figures \
  --runs-dir "$ROOT/runs_final" \
  --output-dir "$ROOT/runs_final/figures"
