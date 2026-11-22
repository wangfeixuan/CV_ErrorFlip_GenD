#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for seed in 2026 2027 2028; do
  "$PYTHON" -m src.train_mixed \
    --blur-features "$ROOT/artifacts/features_blur_k9.npz" \
    --jpeg-features "$ROOT/artifacts/features_jpeg_q30.npz" \
    --resize-features "$ROOT/artifacts/features_resize_25.npz" \
    --type-strategy all_types \
    --seed "$seed" \
    --output-dir "$ROOT/runs_final/mixed_full_seed${seed}" \
    --device cuda

  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_blur_k9.npz" \
    --checkpoint "$ROOT/runs_final/mixed_full_seed${seed}/best.pt" \
    --split test --condition blur_k9 \
    --output "$ROOT/runs_final/mixed_full_seed${seed}/eval_test.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_jpeg_q30.npz" \
    --checkpoint "$ROOT/runs_final/mixed_full_seed${seed}/best.pt" \
    --split test --condition jpeg_q30 \
    --output "$ROOT/runs_final/mixed_full_seed${seed}/eval_test_jpeg_q30.json" \
    --device cuda
  "$PYTHON" -m src.evaluate \
    --features "$ROOT/artifacts/features_resize_25.npz" \
    --checkpoint "$ROOT/runs_final/mixed_full_seed${seed}/best.pt" \
    --split test --condition resize_25 \
    --output "$ROOT/runs_final/mixed_full_seed${seed}/eval_test_resize_25.json" \
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
