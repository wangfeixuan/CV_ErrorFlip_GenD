#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/CV_ErrorFlip_GenD}"
PYTHON="$ROOT/.venv/bin/python"
THREADS="${THREADS:-2}"

cd "$ROOT/code"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

STAGE2_OUT="$ROOT/diagnostics/residual_stage2_final_20261125"
MECHANISM_OUT="$ROOT/diagnostics/adapter_mechanism_stage2_20261125"
FIGURE_OUT="$ROOT/diagnostics/stage2_revision_figures_20261125"

"$PYTHON" -m src.residual_stage2_extended \
  --blur-features "$ROOT/artifacts/features_blur_k9.npz" \
  --jpeg-features "$ROOT/artifacts/features_jpeg_q30.npz" \
  --resize-features "$ROOT/artifacts/features_resize_25.npz" \
  --checkpoint "$ROOT/runs_final/uniform_seed2026/best.pt" \
  --runs-root "$ROOT/runs_final" \
  --seeds 2026 2027 2028 \
  --output-dir "$STAGE2_OUT" \
  --device cpu \
  --bootstrap 10000 \
  --random-repeats 24

"$PYTHON" -m src.adapter_mechanism_stage2 \
  --blur-features "$ROOT/artifacts/features_blur_k9.npz" \
  --jpeg-features "$ROOT/artifacts/features_jpeg_q30.npz" \
  --resize-features "$ROOT/artifacts/features_resize_25.npz" \
  --runs-root "$ROOT/runs_final" \
  --seeds 2026 2027 2028 \
  --output-dir "$MECHANISM_OUT" \
  --device cpu

"$PYTHON" -m src.plot_stage2_revision \
  --stage2-dir "$STAGE2_OUT" \
  --core-dir "$ROOT/diagnostics/residual_stage2_20261125" \
  --mechanism-dir "$MECHANISM_OUT" \
  --output-dir "$FIGURE_OUT"
