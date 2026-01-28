#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV=${CONDA_ENV:-cuda121}
export PYTHONUNBUFFERED=1

# Inputs
SCENES_DIR=${SCENES_DIR:-"${SCRIPT_DIR}/data/scenes"}
TASKS_DIR=${TASKS_DIR:-"${SCRIPT_DIR}/data/CHINGMU_75_rescaled_0800_42_iter1"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/data/aa_side_by_side_test"}
SCENE_ID=${SCENE_ID:-}
MAX_LABELS=${MAX_LABELS:-30}
RESOLUTION_WIDTH=${RESOLUTION_WIDTH:-640}
RESOLUTION_HEIGHT=${RESOLUTION_HEIGHT:-480}
VIDEO_BACKEND=${VIDEO_BACKEND:-cpu}
SH_DEGREE=${SH_DEGREE:--1}

AA_ON_DIR="${OUTPUT_ROOT}/aa_on"
AA_OFF_DIR="${OUTPUT_ROOT}/aa_off"
COMPARE_OUT="${OUTPUT_ROOT}/aa_side_by_side.mp4"
ERROR_LOG_ON="${OUTPUT_ROOT}/aa_on_error.log"
ERROR_LOG_OFF="${OUTPUT_ROOT}/aa_off_error.log"

if [ -z "${SCENE_ID}" ]; then
  if [ ! -d "${TASKS_DIR}" ]; then
    echo "[ERROR] TASKS_DIR does not exist: ${TASKS_DIR}" >&2
    exit 1
  fi
  SCENE_ID=$(ls -1 "${TASKS_DIR}" | sort | awk '/^0001_/{print; exit}')
  if [ -z "${SCENE_ID}" ]; then
    echo "[ERROR] No 0001_* scene found under ${TASKS_DIR}" >&2
    exit 1
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg is required for side-by-side comparison but was not found in PATH." >&2
  exit 1
fi

echo "[CONFIG] SCENE_ID=${SCENE_ID}"
echo "[CONFIG] SCENES_DIR=${SCENES_DIR}"
echo "[CONFIG] TASKS_DIR=${TASKS_DIR}"
echo "[CONFIG] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[CONFIG] MAX_LABELS=${MAX_LABELS}"
echo "[CONFIG] RESOLUTION=${RESOLUTION_WIDTH}x${RESOLUTION_HEIGHT}"

render_one() {
  local aa_flag=$1
  local out_dir=$2
  local err_log=$3
  conda run --no-capture-output -n "${CONDA_ENV}" python "${SCRIPT_DIR}/render_label_paths.py" \
    --scene "${SCENE_ID}" \
    --scenes-dir "${SCENES_DIR}" \
    --tasks-dir "${TASKS_DIR}" \
    --output-dir "${out_dir}" \
    --error-log "${err_log}" \
    --max-labels "${MAX_LABELS}" \
    --video \
    --no-rgb-frames \
    --no-save-depth-maps \
    --save-camera-metadata \
    --no-save-follow-metadata \
    --video-backend "${VIDEO_BACKEND}" \
    --resolution "${RESOLUTION_WIDTH}" "${RESOLUTION_HEIGHT}" \
    --sh-degree "${SH_DEGREE}" \
    "${aa_flag}"
}

render_one --antialiasing "${AA_ON_DIR}" "${ERROR_LOG_ON}"
render_one --no-antialiasing "${AA_OFF_DIR}" "${ERROR_LOG_OFF}"

pick_first_mp4() {
  local root=$1
  find "${root}" -type f -name "*.mp4" | sort | head -n 1
}

AA_ON_VIDEO=$(pick_first_mp4 "${AA_ON_DIR}")
AA_OFF_VIDEO=$(pick_first_mp4 "${AA_OFF_DIR}")

if [ -z "${AA_ON_VIDEO}" ] || [ -z "${AA_OFF_VIDEO}" ]; then
  echo "[ERROR] Could not find mp4 outputs in ${AA_ON_DIR} or ${AA_OFF_DIR}." >&2
  exit 1
fi

ffmpeg -y \
  -i "${AA_ON_VIDEO}" \
  -i "${AA_OFF_VIDEO}" \
  -filter_complex \
    "[0:v]drawtext=text='AA ON':x=12:y=12:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5[v0]; \
     [1:v]drawtext=text='AA OFF':x=12:y=12:fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5[v1]; \
     [v0][v1]hstack=inputs=2[v]" \
  -map "[v]" -map 0:a? -shortest \
  -c:v libx264 -crf 20 -preset medium \
  "${COMPARE_OUT}"

echo "[DONE] Side-by-side video: ${COMPARE_OUT}"
