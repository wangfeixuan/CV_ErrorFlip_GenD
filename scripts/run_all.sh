#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

if [[ ! -f "$ROOT/artifacts/features_blur_k9.npz" ]]; then
  "$PYTHON" -m src.extract_features \
    --original-zip "$ROOT/data/selected_faces.zip" \
    --blur-zip "$ROOT/data/selected_faces_blur_k9.zip" \
    --model "$ROOT/data/model.safetensors" \
    --output "$ROOT/artifacts/features_blur_k9.npz" \
    --device cuda \
    --batch-size 16
fi

bash scripts/run_three_seeds.sh
bash scripts/run_unseen_degradations.sh
bash scripts/run_k_sweep.sh
bash scripts/run_mixed_adapter.sh
bash scripts/run_mixed_full.sh
bash scripts/run_single_degradations.sh
