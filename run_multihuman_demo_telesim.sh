#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

CONDA_ENV=${CONDA_ENV:-cuda121}
USE_CONDA_RUN=${USE_CONDA_RUN:-auto}

SCENE_ID=${SCENE_ID:-0009_858969}
SCENE_NAME=${SCENE_NAME:-${SCENE_ID}}

SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_rescaled_2}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_2_paths_0800}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}

CAMERA_LABEL=${CAMERA_LABEL:-563}
HUMAN_LABELS=${HUMAN_LABELS:-82,97,141,247,330,103,648,1028}
SEED=${SEED:-42}

OUTPUT_ROOT=${OUTPUT_ROOT:-./data1/multihuman_demo}
VIDEO_NAME=${VIDEO_NAME:-}
MANIFEST_NAME=${MANIFEST_NAME:-multihuman_demo.json}

MINIMAL_FRAMES=${MINIMAL_FRAMES:-0}
RESOLUTION_W=${RESOLUTION_W:-960}
RESOLUTION_H=${RESOLUTION_H:-720}
VIDEO_FPS=${VIDEO_FPS:-10}
VIDEO_BACKEND=${VIDEO_BACKEND:-cpu}
ROTATE_180=${ROTATE_180:-true}

OVERWRITE=${OVERWRITE:-true}
DRY_RUN=${DRY_RUN:-false}
MIN_CAMERA_HUMAN_DISTANCE=${MIN_CAMERA_HUMAN_DISTANCE:-1.0}
MIN_HUMAN_HUMAN_DISTANCE=${MIN_HUMAN_HUMAN_DISTANCE:-0}
CAMERA_SPEED_RATIO_MAX=${CAMERA_SPEED_RATIO_MAX:-2.5}
PHASE_SEARCH_TRIALS=${PHASE_SEARCH_TRIALS:-240}
FACING_WINDOW=${FACING_WINDOW:-5}
FACING_EMA_ALPHA=${FACING_EMA_ALPHA:-0.35}
STRICT_HUMAN_HUMAN_DISTANCE=${STRICT_HUMAN_HUMAN_DISTANCE:-false}

cmd=()
if [ "${USE_CONDA_RUN}" = "auto" ]; then
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
    USE_CONDA_RUN="false"
  else
    USE_CONDA_RUN="true"
  fi
fi
if [ "${USE_CONDA_RUN}" = "true" ]; then
  cmd+=(conda run --no-capture-output -n "${CONDA_ENV}")
fi

if [ "${VIDEO_BACKEND}" = "nvenc" ]; then
  nvenc_ok=false
  if [ "${USE_CONDA_RUN}" = "true" ]; then
    if conda run --no-capture-output -n "${CONDA_ENV}" bash -lc "ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'h264_nvenc'"; then
      nvenc_ok=true
    fi
  else
    if command -v ffmpeg >/dev/null 2>&1 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then
      nvenc_ok=true
    fi
  fi
  if [ "${nvenc_ok}" != "true" ]; then
    echo "[WARN] NVENC encoder (h264_nvenc) not available; falling back to VIDEO_BACKEND=cpu." >&2
    VIDEO_BACKEND=cpu
  fi
fi

cmd+=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/render_multihuman_telesim.py"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --scene "${SCENE_ID}"
  --scene-name "${SCENE_NAME}"
  --actor-root "${ACTOR_ROOT}"
  --camera-label "${CAMERA_LABEL}"
  --human-labels "${HUMAN_LABELS}"
  --seed "${SEED}"
  --output-root "${OUTPUT_ROOT}"
  --manifest-name "${MANIFEST_NAME}"
  --resolution "${RESOLUTION_W}" "${RESOLUTION_H}"
  --video-fps "${VIDEO_FPS}"
  --video-backend "${VIDEO_BACKEND}"
  --min-camera-human-distance "${MIN_CAMERA_HUMAN_DISTANCE}"
  --min-human-human-distance "${MIN_HUMAN_HUMAN_DISTANCE}"
  --camera-speed-ratio-max "${CAMERA_SPEED_RATIO_MAX}"
  --phase-search-trials "${PHASE_SEARCH_TRIALS}"
  --facing-window "${FACING_WINDOW}"
  --facing-ema-alpha "${FACING_EMA_ALPHA}"
)
if [ "${STRICT_HUMAN_HUMAN_DISTANCE}" = "true" ]; then
  cmd+=(--strict-human-human-distance)
else
  cmd+=(--no-strict-human-human-distance)
fi
if [ "${ROTATE_180}" = "true" ]; then
  cmd+=(--rotate-180)
else
  cmd+=(--no-rotate-180)
fi

if [ -f "${BAN_LIST}" ]; then
  cmd+=(--ban-list "${BAN_LIST}")
fi
if [ -n "${VIDEO_NAME}" ]; then
  cmd+=(--video-name "${VIDEO_NAME}")
fi
if [ "${MINIMAL_FRAMES}" != "0" ]; then
  cmd+=(--minimal-frames "${MINIMAL_FRAMES}")
fi
if [ "${OVERWRITE}" = "true" ]; then
  cmd+=(--overwrite)
else
  cmd+=(--no-overwrite)
fi
if [ "${DRY_RUN}" = "true" ]; then
  cmd+=(--dry-run)
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "[RUN] scene=${SCENE_ID} camera_label=${CAMERA_LABEL} human_labels=${HUMAN_LABELS}" >&2
echo "[RUN] output_root=${OUTPUT_ROOT}/${SCENE_NAME}" >&2
"${cmd[@]}"
