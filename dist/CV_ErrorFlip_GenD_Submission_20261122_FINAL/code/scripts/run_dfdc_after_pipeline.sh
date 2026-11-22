#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PIPELINE_PID=${1:-}
cd "$ROOT/code"

if [[ -n "$PIPELINE_PID" ]]; then
  while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep 60
  done
fi

for required in \
  "$ROOT/data/dfdc/selected_faces.zip" \
  "$ROOT/data/dfdc/selected_faces_blur_k9.zip" \
  "$ROOT/data/dfdc/selected_faces_jpeg_q30.zip" \
  "$ROOT/data/dfdc/selected_faces_resize_25.zip" \
  "$ROOT/artifacts/dfdc_blur_k9.npz" \
  "$ROOT/artifacts/dfdc_jpeg_q30.npz" \
  "$ROOT/artifacts/dfdc_resize_25.npz"; do
  if [[ ! -f "$required" ]]; then
    echo "Pipeline incomplete; missing $required" >&2
    exit 1
  fi
done

SEEDS=${SEEDS:-"2026 2027 2028"} bash scripts/run_dfdc_experiments.sh
