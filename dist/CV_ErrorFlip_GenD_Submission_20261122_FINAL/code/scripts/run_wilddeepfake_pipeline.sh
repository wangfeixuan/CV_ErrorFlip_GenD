#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/CV_ErrorFlip_GenD
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/code"

$PYTHON -m src.import_wilddeepfake \
  --shard-manifest manifests/wilddeepfake_subset_shards.csv \
  --archives-dir "$ROOT/data/wilddeepfake_subset/archives" \
  --output-zip "$ROOT/data/wilddeepfake_subset/selected_faces.zip" \
  --output-manifest "$ROOT/data/wilddeepfake_subset/selected_sequences.csv"

for kind in blur_k9 jpeg_q30 resize_25; do
  $PYTHON -m src.prepare_degradations \
    --original-zip "$ROOT/data/wilddeepfake_subset/selected_faces.zip" \
    --output-zip "$ROOT/data/wilddeepfake_subset/selected_faces_${kind}.zip" \
    --kind "$kind"
  $PYTHON -m src.extract_features \
    --original-zip "$ROOT/data/wilddeepfake_subset/selected_faces.zip" \
    --blur-zip "$ROOT/data/wilddeepfake_subset/selected_faces_${kind}.zip" \
    --model "$ROOT/data/model.safetensors" \
    --output "$ROOT/artifacts/wilddeepfake_${kind}.npz" \
    --device cuda --batch-size 16
done
