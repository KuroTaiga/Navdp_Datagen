#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN")
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

OUT_ROOT=${OUT_ROOT:-"${GS_DIR}/analysis/lighting_eval"}
REPORT_ROOT=${REPORT_ROOT:-"$OUT_ROOT"}
FRAME_STEP=${FRAME_STEP:-1}
PIXEL_STEP=${PIXEL_STEP:-4}

BASE_DIR="${OUT_ROOT}/base"
POST_DIR="${OUT_ROOT}/post"
RENDER_DIR="${OUT_ROOT}/render"
CL_DIR="${OUT_ROOT}/cl"

BASE_REPORT="${REPORT_ROOT}/base_lighting.json"
POST_REPORT="${REPORT_ROOT}/post_report.json"
POST_LIGHTING="${REPORT_ROOT}/post_lighting.json"
RENDER_REPORT="${REPORT_ROOT}/render_lighting.json"
CL_REPORT="${REPORT_ROOT}/cl_lighting.json"

METRICS_BASE="${REPORT_ROOT}/metrics_base_summary.json"
METRICS_RENDER="${REPORT_ROOT}/metrics_render_summary.json"
METRICS_CL="${REPORT_ROOT}/metrics_cl_summary.json"

if [ -d "$BASE_DIR" ]; then
  echo "[1/5] Base lighting report..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
    "$BASE_DIR" \
    --pattern "*.mp4" \
    --frame-step "$FRAME_STEP" \
    --pixel-step "$PIXEL_STEP" \
    --output-json "$BASE_REPORT"
fi

if [ ! -f "$BASE_REPORT" ]; then
  echo "[ERROR] Base lighting report missing: ${BASE_REPORT}" >&2
  echo "[ERROR] Run the base render first (see run_lighting_variants.sh)." >&2
  exit 1
fi

if [ -d "$POST_DIR" ] && [ ! -f "$POST_REPORT" ]; then
  echo "[2/5] Post lighting report (no filter timing available)..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
    "$POST_DIR" \
    --pattern "*.mp4" \
    --frame-step "$FRAME_STEP" \
    --pixel-step "$PIXEL_STEP" \
    --output-json "$POST_LIGHTING"
fi

if [ -d "$RENDER_DIR" ]; then
  echo "[3/5] Render-time lighting report..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
    "$RENDER_DIR" \
    --pattern "*.mp4" \
    --frame-step "$FRAME_STEP" \
    --pixel-step "$PIXEL_STEP" \
    --output-json "$RENDER_REPORT"
fi

if [ -d "$CL_DIR" ]; then
  echo "[4/5] Camera light report..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
    "$CL_DIR" \
    --pattern "*.mp4" \
    --frame-step "$FRAME_STEP" \
    --pixel-step "$PIXEL_STEP" \
    --output-json "$CL_REPORT"
fi

if [ -d "${OUT_ROOT}/metrics_base" ]; then
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/metrics_summary.py" \
    --metrics-dir "${OUT_ROOT}/metrics_base" \
    --output-json "$METRICS_BASE"
fi
if [ -d "${OUT_ROOT}/metrics_render" ]; then
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/metrics_summary.py" \
    --metrics-dir "${OUT_ROOT}/metrics_render" \
    --output-json "$METRICS_RENDER"
fi
if [ -d "${OUT_ROOT}/metrics_cl" ]; then
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/metrics_summary.py" \
    --metrics-dir "${OUT_ROOT}/metrics_cl" \
    --output-json "$METRICS_CL"
fi

echo "[5/5] Compare lighting reports..."
COMPARE_ARGS=(
  --base-json "$BASE_REPORT"
  --output-json "${REPORT_ROOT}/lighting_compare.json"
  --output-csv "${REPORT_ROOT}/lighting_compare.csv"
)
if [ -f "$POST_REPORT" ]; then
  COMPARE_ARGS+=(--post-json "$POST_REPORT")
elif [ -f "$POST_LIGHTING" ]; then
  COMPARE_ARGS+=(--post-json "$POST_LIGHTING")
fi
if [ -f "$RENDER_REPORT" ]; then
  COMPARE_ARGS+=(--render-json "$RENDER_REPORT" --metrics-render "$METRICS_RENDER")
fi
if [ -f "$CL_REPORT" ]; then
  COMPARE_ARGS+=(--cl-json "$CL_REPORT" --metrics-cl "$METRICS_CL")
fi
if [ -f "$METRICS_BASE" ]; then
  COMPARE_ARGS+=(--metrics-base "$METRICS_BASE")
fi

"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_compare_report.py" "${COMPARE_ARGS[@]}"

echo "Reports written to ${REPORT_ROOT}"
