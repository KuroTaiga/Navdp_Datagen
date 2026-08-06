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
SCENES_DIR=${SCENES_DIR:-./data/scenes}
OUTPUT_DIR=${OUTPUT_DIR:-}
SCENE_ID=${SCENE_ID:-}
MAX_SCENES=${MAX_SCENES:-}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-5.0}
CUT_ABOVE=${CUT_ABOVE:-2.5}
CEILING_EPSILON=${CEILING_EPSILON:-0.02}
BELOW_FLOOR_TOLERANCE=${BELOW_FLOOR_TOLERANCE:-0.05}
PADDING=${PADDING:-0.1}
WIDTH=${WIDTH:-512}
HEIGHT=${HEIGHT:-512}
ROTATE_K=${ROTATE_K:--1}
SH_DEGREE=${SH_DEGREE:-3}
MODE=${MODE:-render}
POINT_SIZE=${POINT_SIZE:-2}
ANTIALIASING=${ANTIALIASING:-true}
OVERWRITE=${OVERWRITE:-true}
ALPHA=${ALPHA:-false}
BG_R=${BG_R:-1.0}
BG_G=${BG_G:-1.0}
BG_B=${BG_B:-1.0}

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
  "$PYTHON_BIN" "$SCRIPT_DIR/render_room_topdown_views.py"
  --scenes-dir "${SCENES_DIR}"
  --camera-height "${CAMERA_HEIGHT}"
  --cut-above "${CUT_ABOVE}"
  --ceiling-epsilon "${CEILING_EPSILON}"
  --below-floor-tolerance "${BELOW_FLOOR_TOLERANCE}"
  --padding "${PADDING}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --rotate-k "${ROTATE_K}"
  --sh-degree "${SH_DEGREE}"
  --mode "${MODE}"
  --point-size "${POINT_SIZE}"
  --bg-color "${BG_R}" "${BG_G}" "${BG_B}"
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
if [ "${ALPHA}" = "true" ]; then
  cmd+=(--alpha)
fi

echo "[INFO] Top-down BEV plane is x-y; z is vertical." >&2
echo "[RUN] ${cmd[*]}" >&2
exec "${cmd[@]}"
