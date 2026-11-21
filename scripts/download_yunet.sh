#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
OUTPUT="$ROOT/data/yunet_face_detection_yunet_2023mar.onnx"
EXPECTED_SHA256=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
URL=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

mkdir -p "$(dirname "$OUTPUT")"
if [[ ! -f "$OUTPUT" ]]; then
  curl -L --fail --retry 3 --output "$OUTPUT.tmp" "$URL"
  mv "$OUTPUT.tmp" "$OUTPUT"
fi

ACTUAL_SHA256=$(shasum -a 256 "$OUTPUT" | awk '{print $1}')
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "YuNet checksum mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi
echo "$OUTPUT"
