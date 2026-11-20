#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for k in 1 2 3 4; do
  for seed in 2026 2027 2028; do
    "$PYTHON" -m src.train \
      --features "$ROOT/artifacts/features_blur_k9.npz" \
      --method uniform_random \
      --blurred-frames "$k" \
      --seed "$seed" \
      --output-dir "$ROOT/runs_final/k${k}_random_seed${seed}" \
      --device cuda

    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_blur_k9.npz" \
      --checkpoint "$ROOT/runs_final/k${k}_random_seed${seed}/best.pt" \
      --split test \
      --condition blur_k9 \
      --output "$ROOT/runs_final/k${k}_random_seed${seed}/eval_test.json" \
      --device cuda

    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_jpeg_q30.npz" \
      --checkpoint "$ROOT/runs_final/k${k}_random_seed${seed}/best.pt" \
      --split test \
      --condition jpeg_q30 \
      --output "$ROOT/runs_final/k${k}_random_seed${seed}/eval_test_jpeg_q30.json" \
      --device cuda

    "$PYTHON" -m src.evaluate \
      --features "$ROOT/artifacts/features_resize_25.npz" \
      --checkpoint "$ROOT/runs_final/k${k}_random_seed${seed}/best.pt" \
      --split test \
      --condition resize_25 \
      --output "$ROOT/runs_final/k${k}_random_seed${seed}/eval_test_resize_25.json" \
      --device cuda
  done
done

"$PYTHON" -m src.aggregate_results \
  --runs-dir "$ROOT/runs_final" \
  --output-prefix "$ROOT/runs_final/aggregate_all_conditions"
"$PYTHON" -m src.plot_k_sweep \
  --runs-dir "$ROOT/runs_final" \
  --output-prefix "$ROOT/runs_final/k_sweep"
"$PYTHON" -m src.compare_k_models \
  --runs-dir "$ROOT/runs_final" \
  --output "$ROOT/runs_final/k_sweep_bootstrap.csv" \
  --reference-k 2
