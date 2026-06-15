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

# User-configurable defaults (override via env vars)
CONDA_ENV=${CONDA_ENV:-cuda121}
USE_CONDA_RUN=${USE_CONDA_RUN:-auto}
SCENES_DIR=${SCENES_DIR:-./data/scenes}
OUTPUT_DIR=${OUTPUT_DIR:-}
SCENE_ID=${SCENE_ID:-}
MAX_SCENES=${MAX_SCENES:-}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-1.5}
WIDTH=${WIDTH:-512}
HEIGHT=${HEIGHT:-512}
FOV_DEG=${FOV_DEG:-90}
ZNEAR=${ZNEAR:-0.001}
ZFAR=${ZFAR:-30}
LOOK_DOWN=${LOOK_DOWN:-0}
START_YAW_DEG=${START_YAW_DEG:-0}
ROTATE_K=${ROTATE_K:-2}
SH_DEGREE=${SH_DEGREE:-3}
ANTIALIASING=${ANTIALIASING:-false}
OVERWRITE=${OVERWRITE:-false}

cmd=()
if [ "${USE_CONDA_RUN}" = "auto" ]; then
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
    USE_CONDA_RUN="false"
  else
    USE_CONDA_RUN="true"
  fi
fi

if [ "${USE_CONDA_RUN}" = "true" ]; then
  cmd+=(conda run --no-capture-output -n "$CONDA_ENV")
fi

cmd+=(
  "$PYTHON_BIN" "$SCRIPT_DIR/render_room_center_views.py"
  --scenes-dir "${SCENES_DIR}"
  --camera-height "${CAMERA_HEIGHT}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --fov-deg "${FOV_DEG}"
  --znear "${ZNEAR}"
  --zfar "${ZFAR}"
  --look-down "${LOOK_DOWN}"
  --start-yaw-deg "${START_YAW_DEG}"
  --rotate-k "${ROTATE_K}"
  --sh-degree "${SH_DEGREE}"
)

if [ -n "${OUTPUT_DIR}" ]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${SCENE_ID}" ]; then
  cmd+=(--scene "${SCENE_ID}")
fi
if [ -n "${MAX_SCENES}" ]; then
  cmd+=(--max-scenes "${MAX_SCENES}")
fi
if [ "${ANTIALIASING}" = "true" ]; then
  cmd+=(--antialiasing)
fi
if [ "${OVERWRITE}" = "true" ]; then
  cmd+=(--overwrite)
fi

echo "[INFO] BEV/floor plane is x-y; z is vertical." >&2
echo "[RUN] ${cmd[*]}" >&2
exec "${cmd[@]}"
