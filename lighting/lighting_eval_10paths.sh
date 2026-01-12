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

TASKS_DIR=${TASKS_DIR:-"${GS_DIR}/data/interiorGS_0500_42"}
SCENES_DIR=${SCENES_DIR:-"${GS_DIR}/data/scenes"}
OUT_ROOT=${OUT_ROOT:-"${GS_DIR}/analysis/lighting_eval"}
COUNT=${COUNT:-10}
SEED=${SEED:-12345}

LIGHT_MODE=${LIGHT_MODE:-disk}
LIGHT_STRENGTH=${LIGHT_STRENGTH:-0.30}
LIGHT_RADIUS=${LIGHT_RADIUS:-0.45}
LIGHT_CENTER_X=${LIGHT_CENTER_X:-0.5}
LIGHT_CENTER_Y=${LIGHT_CENTER_Y:-0.5}
LIGHT_JITTER=${LIGHT_JITTER:-0.0}
LIGHT_TEMP_K=${LIGHT_TEMP_K:-3000}
LIGHT_VIGNETTE=${LIGHT_VIGNETTE:-0.20}
LIGHT_SEED=${LIGHT_SEED:-123}

COMMON_RENDER_ARGS="--view-mode forward --gpu-only --height-offset 0.3 --no-rgb-frames --no-save-depth-maps --no-save-camera-metadata --no-show-BEV"

SAMPLE_JSON="${OUT_ROOT}/sample_${COUNT}.json"
SAMPLE_TXT="${OUT_ROOT}/sample_${COUNT}.txt"

mkdir -p "$OUT_ROOT"

echo "[1/5] Sampling ${COUNT} paths..."
"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/sample_paths.py" \
  --tasks-dir "$TASKS_DIR" \
  --count "$COUNT" \
  --seed "$SEED" \
  --output-json "$SAMPLE_JSON" \
  --output-txt "$SAMPLE_TXT"

echo "[2/5] Base render (no lighting filter)..."
"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
  --sample-json "$SAMPLE_JSON" \
  --scenes-dir "$SCENES_DIR" \
  --tasks-dir "$TASKS_DIR" \
  --output-dir "${OUT_ROOT}/base" \
  --metrics-dir "${OUT_ROOT}/metrics_base" \
  --render-extra-args "$COMMON_RENDER_ARGS"

echo "[3/5] Base lighting report..."
"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
  "${OUT_ROOT}/base" \
  --pattern "*.mp4" \
  --frame-step 1 \
  --pixel-step 4 \
  --output-json "${OUT_ROOT}/base_lighting.json"

echo "[4/5] Post-process MP4 lighting + report..."
"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/apply_light_filter_mp4.py" \
  "${OUT_ROOT}/base" \
  --pattern "*.mp4" \
  --output-dir "${OUT_ROOT}/post" \
  --suffix "_${LIGHT_MODE}${LIGHT_TEMP_K}" \
  --light-mode "$LIGHT_MODE" \
  --light-strength "$LIGHT_STRENGTH" \
  --light-radius "$LIGHT_RADIUS" \
  --light-center "$LIGHT_CENTER_X" "$LIGHT_CENTER_Y" \
  --light-jitter "$LIGHT_JITTER" \
  --light-temp-k "$LIGHT_TEMP_K" \
  --light-vignette "$LIGHT_VIGNETTE" \
  --light-seed "$LIGHT_SEED" \
  --output-json "${OUT_ROOT}/post_report.json"

echo "[5/5] Render-time lighting + report..."
"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
  --sample-json "$SAMPLE_JSON" \
  --scenes-dir "$SCENES_DIR" \
  --tasks-dir "$TASKS_DIR" \
  --output-dir "${OUT_ROOT}/render" \
  --metrics-dir "${OUT_ROOT}/metrics_render" \
  --render-extra-args "${COMMON_RENDER_ARGS} --light-mode ${LIGHT_MODE} --light-strength ${LIGHT_STRENGTH} --light-radius ${LIGHT_RADIUS} --light-center ${LIGHT_CENTER_X} ${LIGHT_CENTER_Y} --light-jitter ${LIGHT_JITTER} --light-temp-k ${LIGHT_TEMP_K} --light-vignette ${LIGHT_VIGNETTE} --light-seed ${LIGHT_SEED}"

"${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
  "${OUT_ROOT}/render" \
  --pattern "*.mp4" \
  --frame-step 1 \
  --pixel-step 4 \
  --output-json "${OUT_ROOT}/render_lighting.json"

echo "Done."
echo "Base lighting:    ${OUT_ROOT}/base_lighting.json"
echo "Post report:      ${OUT_ROOT}/post_report.json"
echo "Render lighting:  ${OUT_ROOT}/render_lighting.json"
echo "Metrics (base):   ${OUT_ROOT}/metrics_base"
echo "Metrics (render): ${OUT_ROOT}/metrics_render"
