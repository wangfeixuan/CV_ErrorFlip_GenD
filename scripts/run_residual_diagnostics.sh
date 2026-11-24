#!/usr/bin/env bash
set -euo pipefail

ROOT=${CV_ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}
PYTHON="$ROOT/.venv/bin/python"
OUTPUT=${RESIDUAL_OUTPUT:-$ROOT/diagnostics/residual_20261125}
CHECKPOINT=${ADAPTER_CHECKPOINT:-$ROOT/runs_final/uniform_seed2026/best.pt}
cd "$ROOT/code"

"$PYTHON" -m src.residual_diagnostics \
  --blur-features "$ROOT/artifacts/features_blur_k9.npz" \
  --jpeg-features "$ROOT/artifacts/features_jpeg_q30.npz" \
  --resize-features "$ROOT/artifacts/features_resize_25.npz" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --device "${RESIDUAL_DEVICE:-cpu}"
