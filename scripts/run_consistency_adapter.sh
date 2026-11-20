#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for seed in 2026 2027 2028; do
  run_dir="$ROOT/runs_final/consistency_seed${seed}"
  "$PYTHON" -m src.train \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --method consistency \
    --seed "$seed" \
    --output-dir "$run_dir" \
    --device cuda

  for condition in blur_k9 jpeg_q30 resize_25; do
    case "$condition" in
      blur_k9)
        feature_file="$ROOT/artifacts/features_blur_k9.npz"
        output="$run_dir/eval_test.json"
        ;;
      jpeg_q30)
        feature_file="$ROOT/artifacts/features_jpeg_q30.npz"
        output="$run_dir/eval_test_jpeg_q30.json"
        ;;
      resize_25)
        feature_file="$ROOT/artifacts/features_resize_25.npz"
        output="$run_dir/eval_test_resize_25.json"
        ;;
    esac
    "$PYTHON" -m src.evaluate \
      --features "$feature_file" \
      --checkpoint "$run_dir/best.pt" \
      --split test --condition "$condition" \
      --output "$output" \
      --device cuda
  done
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
