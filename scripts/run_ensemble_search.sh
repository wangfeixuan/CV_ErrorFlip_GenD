#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for seed in 2026 2027 2028; do
  for method in clean_only uniform jpeg_only resize_only; do
    run_dir="$ROOT/runs_final/${method}_seed${seed}"
    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_blur_k9.npz" \
      --checkpoint "$run_dir/best.pt" \
      --split validation --condition blur_k9 \
      --output "$run_dir/eval_validation.json" \
      --device cuda
    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_jpeg_q30.npz" \
      --checkpoint "$run_dir/best.pt" \
      --split validation --condition jpeg_q30 \
      --output "$run_dir/eval_validation_jpeg_q30.json" \
      --device cuda
    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_resize_25.npz" \
      --checkpoint "$run_dir/best.pt" \
      --split validation --condition resize_25 \
      --output "$run_dir/eval_validation_resize_25.json" \
      --device cuda
  done
done

"$PYTHON" -m src.search_ensemble_weights \
  --runs-dir "$ROOT/runs_final" \
  --output-prefix "$ROOT/runs_final/ensemble_weight_search" \
  --blur-step 0.1 \
  --clean-weights 0,0.1,0.2,0.3
