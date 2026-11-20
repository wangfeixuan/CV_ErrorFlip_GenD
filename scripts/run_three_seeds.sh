#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
FEATURES="$ROOT/artifacts/features_blur_k9.npz"
RUN_ROOT="$ROOT/runs_final"
PYTHON="$ROOT/.venv/bin/python"

for seed in 2026 2027 2028; do
  for method in uniform flip_weighted flip_hard; do
    "$PYTHON" -m src.train \
      --features "$FEATURES" \
      --method "$method" \
      --seed "$seed" \
      --output-dir "$RUN_ROOT/${method}_seed${seed}" \
      --device cuda
    "$PYTHON" -m src.evaluate \
      --features "$FEATURES" \
      --checkpoint "$RUN_ROOT/${method}_seed${seed}/best.pt" \
      --split test \
      --output "$RUN_ROOT/${method}_seed${seed}/eval_test.json" \
      --device cuda
  done
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$RUN_ROOT" \
  --output-prefix "$RUN_ROOT/aggregate_test"
