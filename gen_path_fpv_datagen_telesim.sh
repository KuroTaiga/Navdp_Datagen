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

# Defaults mirror run_random_fpv_datagen_telesim.sh
SCENE_ID=${SCENE_ID:-}
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_75_scenes}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./CHINGMU_0800}
MAX_LABELS=${MAX_LABELS:-}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-0}
FPV_FOLLOW_DISTANCE=${FPV_FOLLOW_DISTANCE:-0}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}
WORKERS=${WORKERS:-16}
TAR_ZST=${TAR_ZST:-false}
ZSTD_LEVEL=${ZSTD_LEVEL:-3}

# Two separate dataset roots (preserve <dataset>/<scene>/...)
CAMERA_DATASET_DIR=${CAMERA_DATASET_DIR:-${OUTPUT_DIR}_camera}
ACTIONS_DATASET_DIR=${ACTIONS_DATASET_DIR:-${OUTPUT_DIR}_actions}

# FPV does NOT need an actor assignment manifest.
cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/datasets/gen_path_dataset_telesim.py"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --camera-root "${CAMERA_DATASET_DIR}"
  --actions-root "${ACTIONS_DATASET_DIR}"
  --follow-distance "${FPV_FOLLOW_DISTANCE}"
  --height-offset "${HEIGHT_OFFSET}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --scene-workers "${WORKERS}"
  --overwrite
)

if [ -n "${SCENE_ID}" ]; then
  cmd+=(--scene "${SCENE_ID}")
fi
if [ -n "${MAX_LABELS}" ]; then
  cmd+=(--max-labels "${MAX_LABELS}")
fi
if [ "${EXCLUDE_DETAILED_LABELS}" = "true" ]; then
  cmd+=(--exclude-detailed-labels)
else
  cmd+=(--no-exclude-detailed-labels)
fi

echo "[GEN] camera_root=${CAMERA_DATASET_DIR}" >&2
echo "[GEN] actions_root=${ACTIONS_DATASET_DIR}" >&2
echo "[GEN] scenes_dir=${SCENES_DIR}" >&2
echo "[GEN] tasks_dir=${TASKS_DIR}" >&2

if [ -n "${TAR_OUT:-}" ]; then
  cmd+=(--tar-out "${TAR_OUT}")
fi
if [ "${TAR_ZST}" = "true" ]; then
  TAR_CAMERA_OUT=${TAR_CAMERA_OUT:-${CAMERA_DATASET_DIR}.tar.zst}
  TAR_ACTIONS_OUT=${TAR_ACTIONS_OUT:-${ACTIONS_DATASET_DIR}.tar.zst}
  cmd+=(--tar-camera-zst --tar-camera-out "${TAR_CAMERA_OUT}")
  cmd+=(--tar-actions-zst --tar-actions-out "${TAR_ACTIONS_OUT}")
  cmd+=(--zstd-level "${ZSTD_LEVEL}")
fi

"${cmd[@]}"
