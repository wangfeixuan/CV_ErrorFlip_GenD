#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

for condition in jpeg_q30 resize_25; do
  if [[ ! -f "$ROOT/data/selected_faces_${condition}.zip" ]]; then
    "$PYTHON" -m src.prepare_degradations \
      --original-zip "$ROOT/data/selected_faces.zip" \
      --output-zip "$ROOT/data/selected_faces_${condition}.zip" \
      --kind "$condition"
  fi
  if [[ ! -f "$ROOT/artifacts/features_${condition}.npz" ]]; then
    "$PYTHON" -m src.extract_features \
      --original-zip "$ROOT/data/selected_faces.zip" \
      --blur-zip "$ROOT/data/selected_faces_${condition}.zip" \
      --model "$ROOT/data/model.safetensors" \
      --output "$ROOT/artifacts/features_${condition}.npz" \
      --device cuda \
      --batch-size 16
  fi

  for seed in 2026 2027 2028; do
    for method in clean_only uniform flip_weighted flip_hard; do
      "$PYTHON" -m src.evaluate \
        --features "$ROOT/artifacts/features_${condition}.npz" \
        --checkpoint "$ROOT/runs_final/${method}_seed${seed}/best.pt" \
        --split test \
        --condition "$condition" \
        --output "$ROOT/runs_final/${method}_seed${seed}/eval_test_${condition}.json" \
        --device cuda
    done
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
