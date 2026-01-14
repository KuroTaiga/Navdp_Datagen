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

bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

TASKS_DIR=${TASKS_DIR:-"${GS_DIR}/data/interiorGS_0500_42"}
SCENES_DIR=${SCENES_DIR:-"${GS_DIR}/data/scenes"}
OUT_ROOT=${OUT_ROOT:-"${GS_DIR}/analysis/lighting_eval"}
COUNT=${COUNT:-10}
SEED=${SEED:-12345}
REUSE_SAMPLE=${REUSE_SAMPLE:-true}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}

DO_BASE=${DO_BASE:-true}
DO_POST=${DO_POST:-true}
DO_RENDER=${DO_RENDER:-true}
DO_CL=${DO_CL:-true}

COMMON_RENDER_ARGS=${COMMON_RENDER_ARGS:-"--view-mode forward --gpu-only --height-offset ${HEIGHT_OFFSET} --no-rgb-frames --no-save-depth-maps --no-save-camera-metadata --no-show-BEV"}

# Post-process settings (lower lighting by default).
POST_MODE=${POST_MODE:-global}
POST_STRENGTH=${POST_STRENGTH:--0.5}
POST_RADIUS=${POST_RADIUS:-0.45}
POST_CENTER_X=${POST_CENTER_X:-0.5}
POST_CENTER_Y=${POST_CENTER_Y:-0.5}
POST_JITTER=${POST_JITTER:-0.0}
POST_TEMP_K=${POST_TEMP_K:-0}
POST_VIGNETTE=${POST_VIGNETTE:-0.0}
POST_SEED=${POST_SEED:-123}

# Render-time filter settings (lower lighting by default).
RENDER_LIGHT_MODE=${RENDER_LIGHT_MODE:-global}
RENDER_LIGHT_STRENGTH=${RENDER_LIGHT_STRENGTH:--0.5}
RENDER_LIGHT_RADIUS=${RENDER_LIGHT_RADIUS:-0.45}
RENDER_LIGHT_CENTER_X=${RENDER_LIGHT_CENTER_X:-0.5}
RENDER_LIGHT_CENTER_Y=${RENDER_LIGHT_CENTER_Y:-0.5}
RENDER_LIGHT_JITTER=${RENDER_LIGHT_JITTER:-0.0}
RENDER_LIGHT_TEMP_K=${RENDER_LIGHT_TEMP_K:-0}
RENDER_LIGHT_VIGNETTE=${RENDER_LIGHT_VIGNETTE:-0.0}
RENDER_LIGHT_SEED=${RENDER_LIGHT_SEED:-123}

# Camera light (CL) settings.
CL_SCENES_DIR=${CL_SCENES_DIR:-"$SCENES_DIR"}
CL_STRENGTH=${CL_STRENGTH:-1.0}
CL_COLOR_R=${CL_COLOR_R:-1.0}
CL_COLOR_G=${CL_COLOR_G:-1.0}
CL_COLOR_B=${CL_COLOR_B:-1.0}
CL_AMBIENT=${CL_AMBIENT:-0.2}
CL_DIFFUSE=${CL_DIFFUSE:-1.0}
CL_SPECULAR=${CL_SPECULAR:-0.2}
CL_SHININESS=${CL_SHININESS:-16.0}
CL_RANGE=${CL_RANGE:-0.0}
CL_OFFSET_X=${CL_OFFSET_X:-0.0}
CL_OFFSET_Y=${CL_OFFSET_Y:-0.0}
# Camera coords: X right, Y up, Z forward. Use 0/0/0 to attach to the camera.
CL_OFFSET_Z=${CL_OFFSET_Z:-0.0}
CL_NORMAL_SMOOTH=${CL_NORMAL_SMOOTH:-2}
CL_SHADOW=${CL_SHADOW:-false}
CL_SHADOW_BIAS=${CL_SHADOW_BIAS:-0.02}
CL_SHADOW_STRENGTH=${CL_SHADOW_STRENGTH:-0.2}
CL_SHADOW_PCF=${CL_SHADOW_PCF:-1}

SAMPLE_JSON="${OUT_ROOT}/sample_${COUNT}.json"
SAMPLE_TXT="${OUT_ROOT}/sample_${COUNT}.txt"

mkdir -p "$OUT_ROOT"

if ! bool_true "$REUSE_SAMPLE" || [ ! -f "$SAMPLE_JSON" ]; then
  echo "[1/5] Sampling ${COUNT} paths..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/sample_paths.py" \
    --tasks-dir "$TASKS_DIR" \
    --count "$COUNT" \
    --seed "$SEED" \
    --output-json "$SAMPLE_JSON" \
    --output-txt "$SAMPLE_TXT"
else
  echo "[1/5] Reusing sample list: ${SAMPLE_JSON}"
fi

if bool_true "$DO_BASE"; then
  echo "[2/5] Base render (no lighting filter)..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
    --sample-json "$SAMPLE_JSON" \
    --scenes-dir "$SCENES_DIR" \
    --tasks-dir "$TASKS_DIR" \
    --output-dir "${OUT_ROOT}/base" \
    --metrics-dir "${OUT_ROOT}/metrics_base" \
    --render-extra-args "$COMMON_RENDER_ARGS"
fi

if bool_true "$DO_POST"; then
  echo "[3/5] Post-process MP4 lighting..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/apply_light_filter_mp4.py" \
    "${OUT_ROOT}/base" \
    --pattern "*.mp4" \
    --output-dir "${OUT_ROOT}/post" \
    --suffix "_${POST_MODE}" \
    --light-mode "$POST_MODE" \
    --light-strength "$POST_STRENGTH" \
    --light-radius "$POST_RADIUS" \
    --light-center "$POST_CENTER_X" "$POST_CENTER_Y" \
    --light-jitter "$POST_JITTER" \
    --light-temp-k "$POST_TEMP_K" \
    --light-vignette "$POST_VIGNETTE" \
    --light-seed "$POST_SEED" \
    --output-json "${OUT_ROOT}/post_report.json"
fi

if bool_true "$DO_RENDER"; then
  echo "[4/5] Render-time lighting filter..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
    --sample-json "$SAMPLE_JSON" \
    --scenes-dir "$SCENES_DIR" \
    --tasks-dir "$TASKS_DIR" \
    --output-dir "${OUT_ROOT}/render" \
    --metrics-dir "${OUT_ROOT}/metrics_render" \
    --render-extra-args "${COMMON_RENDER_ARGS} --light-mode ${RENDER_LIGHT_MODE} --light-strength ${RENDER_LIGHT_STRENGTH} --light-radius ${RENDER_LIGHT_RADIUS} --light-center ${RENDER_LIGHT_CENTER_X} ${RENDER_LIGHT_CENTER_Y} --light-jitter ${RENDER_LIGHT_JITTER} --light-temp-k ${RENDER_LIGHT_TEMP_K} --light-vignette ${RENDER_LIGHT_VIGNETTE} --light-seed ${RENDER_LIGHT_SEED}"
fi

if bool_true "$DO_CL"; then
  echo "[5/5] Render-time camera light (CL)..."
  CL_ARGS="--cl-enable --cl-strength ${CL_STRENGTH} --cl-color ${CL_COLOR_R} ${CL_COLOR_G} ${CL_COLOR_B} --cl-ambient ${CL_AMBIENT} --cl-diffuse ${CL_DIFFUSE} --cl-specular ${CL_SPECULAR} --cl-shininess ${CL_SHININESS} --cl-range ${CL_RANGE} --cl-offset ${CL_OFFSET_X} ${CL_OFFSET_Y} ${CL_OFFSET_Z} --cl-normal-smooth ${CL_NORMAL_SMOOTH} --cl-shadow-bias ${CL_SHADOW_BIAS} --cl-shadow-strength ${CL_SHADOW_STRENGTH} --cl-shadow-pcf ${CL_SHADOW_PCF}"
  if bool_true "$CL_SHADOW"; then
    CL_ARGS="${CL_ARGS} --cl-shadow"
  fi
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
    --sample-json "$SAMPLE_JSON" \
    --scenes-dir "$CL_SCENES_DIR" \
    --tasks-dir "$TASKS_DIR" \
    --output-dir "${OUT_ROOT}/cl" \
    --metrics-dir "${OUT_ROOT}/metrics_cl" \
    --render-extra-args "${COMMON_RENDER_ARGS} ${CL_ARGS}"
fi

echo "Done. Outputs in ${OUT_ROOT}"
