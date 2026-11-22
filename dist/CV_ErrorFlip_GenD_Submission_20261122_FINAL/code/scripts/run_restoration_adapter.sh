#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for seed in 2026 2027 2028; do
  run_dir="$ROOT/runs_final/restore_seed${seed}"
  "$PYTHON" -m src.train \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --method restore \
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
"$PYTHON" -m src.summarize_restoration \
  --runs-dir "$ROOT/runs_final" \
  --output "$ROOT/runs_final/restoration_analysis.csv"
