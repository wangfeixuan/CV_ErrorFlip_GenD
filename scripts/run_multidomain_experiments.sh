#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PYTHON="$ROOT/.venv/bin/python"
CONFIG="$ROOT/code/configs/multidomain.yaml"
CELEB_BLUR="$ROOT/artifacts/features_blur_k9.npz"
CELEB_JPEG="$ROOT/artifacts/features_jpeg_q30.npz"
CELEB_RESIZE="$ROOT/artifacts/features_resize_25.npz"
DFDC_BLUR="$ROOT/artifacts/dfdc_blur_k9.npz"
DFDC_JPEG="$ROOT/artifacts/dfdc_jpeg_q30.npz"
DFDC_RESIZE="$ROOT/artifacts/dfdc_resize_25.npz"
RUN_ROOT="$ROOT/runs_multidomain"
SEEDS=${SEEDS:-"2026 2027 2028"}
cd "$ROOT/code"

evaluate_domains() {
  local run_dir="$1"
  "$PYTHON" -m src.evaluate \
    --features "$CELEB_BLUR" --checkpoint "$run_dir/best.pt" \
    --split test --condition blur_k9 \
    --output "$run_dir/eval_celebdf_blur_k9.json" --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$CELEB_JPEG" --checkpoint "$run_dir/best.pt" \
    --split test --condition jpeg_q30 \
    --output "$run_dir/eval_celebdf_jpeg_q30.json" --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$CELEB_RESIZE" --checkpoint "$run_dir/best.pt" \
    --split test --condition resize_25 \
    --output "$run_dir/eval_celebdf_resize_25.json" --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$DFDC_BLUR" --checkpoint "$run_dir/best.pt" \
    --split test --condition blur_k9 \
    --output "$run_dir/eval_dfdc_blur_k9.json" --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$DFDC_JPEG" --checkpoint "$run_dir/best.pt" \
    --split test --condition jpeg_q30 \
    --output "$run_dir/eval_dfdc_jpeg_q30.json" --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$DFDC_RESIZE" --checkpoint "$run_dir/best.pt" \
    --split test --condition resize_25 \
    --output "$run_dir/eval_dfdc_resize_25.json" --device cuda
}

train_variant() {
  local method="$1"
  local identity_weight="$2"
  local seed="$3"
  local label="$4"
  local run_dir="$RUN_ROOT/${label}_seed${seed}"
  "$PYTHON" -m src.train_multidomain \
    --celeb-features "$CELEB_BLUR" \
    --dfdc-features "$DFDC_BLUR" \
    --method "$method" \
    --variant-name "$label" \
    --identity-reg-weight "$identity_weight" \
    --seed "$seed" \
    --output-dir "$run_dir" \
    --config "$CONFIG" \
    --device cuda
  evaluate_domains "$run_dir"
}

for seed in $SEEDS; do
  train_variant clean_only 0.0 "$seed" multidomain_clean
  train_variant uniform 0.0 "$seed" multidomain_uniform
  train_variant uniform 0.05 "$seed" multidomain_uniform_identity
  train_variant flip_weighted 0.05 "$seed" multidomain_flip_identity
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_celebdf" \
  --glob "*/eval_celebdf_*.json"
"$PYTHON" -m src.aggregate_results \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_dfdc" \
  --glob "*/eval_dfdc_*.json"
"$PYTHON" -m src.bootstrap_analysis \
  --runs-dir "$RUN_ROOT" \
  --output "$RUN_ROOT/bootstrap_celebdf.csv" \
  --glob "*/eval_celebdf_*.per_video.csv" \
  --group-manifest "$ROOT/runs_final/clean_only_seed2026/split_manifest.csv" \
  --group-split test \
  --baseline-method multidomain_uniform \
  --clean-method multidomain_clean
