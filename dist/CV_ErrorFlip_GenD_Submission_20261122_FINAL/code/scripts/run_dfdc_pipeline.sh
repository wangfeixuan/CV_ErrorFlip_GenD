#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PYTHON="$ROOT/.venv/bin/python"
DATA="$ROOT/data/dfdc"
VIDEOS=${DFDC_VIDEOS:-$DATA/archive.zip}
METADATA=${DFDC_METADATA:-$DATA/metadata}
YUNET=${YUNET_MODEL:-$ROOT/data/yunet_face_detection_yunet_2023mar.onnx}
DFDC_NUM_THREADS=${DFDC_NUM_THREADS:-2}
cd "$ROOT/code"

mkdir -p "$DATA"
if [[ ! -f "$METADATA" && -f "$DATA/metadata.zip" ]]; then
  METADATA="$DATA/metadata.zip"
fi
if [[ ! -f "$YUNET" ]]; then
  bash scripts/download_yunet.sh
fi
if [[ ! -f "$VIDEOS" ]]; then
  echo "DFDC video archive not found: $VIDEOS" >&2
  exit 1
fi
if [[ ! -f "$METADATA" ]]; then
  echo "DFDC metadata file not found: $METADATA" >&2
  echo "Download the 43 MB metadata dataset from Kaggle and copy it to this path." >&2
  exit 1
fi

if [[ ! -f "$DATA/selected_faces.zip" ]]; then
  "$PYTHON" -m src.import_dfdc \
    --videos "$VIDEOS" \
    --metadata "$METADATA" \
    --yunet-model "$YUNET" \
    --output-zip "$DATA/selected_faces.zip" \
    --output-manifest "$DATA/selected_faces.manifest.csv" \
    --failed-csv "$DATA/failed_videos.csv" \
    --num-threads "$DFDC_NUM_THREADS"
fi

for kind in blur_k9 jpeg_q30 resize_25; do
  if [[ ! -f "$DATA/selected_faces_${kind}.zip" ]]; then
    "$PYTHON" -m src.prepare_degradations \
      --original-zip "$DATA/selected_faces.zip" \
      --output-zip "$DATA/selected_faces_${kind}.zip" \
      --kind "$kind"
  fi
done

for kind in blur_k9 jpeg_q30 resize_25; do
  if [[ ! -f "$ROOT/artifacts/dfdc_${kind}.npz" ]]; then
    "$PYTHON" -m src.extract_features \
      --original-zip "$DATA/selected_faces.zip" \
      --blur-zip "$DATA/selected_faces_${kind}.zip" \
      --group-manifest "$DATA/selected_faces.manifest.csv" \
      --model "$ROOT/data/model.safetensors" \
      --output "$ROOT/artifacts/dfdc_${kind}.npz" \
      --device cuda \
      --batch-size 16
  fi
done
