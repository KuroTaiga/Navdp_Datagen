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

# Defaults mirror run_random_human_datagen_telesim.sh
CONDA_ENV=${CONDA_ENV:-cuda121}
USE_CONDA_RUN=${USE_CONDA_RUN:-auto}
SEED=${SEED:-1}
SCENE_ID=${SCENE_ID:-}
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_scenes_rescaled}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./navdata/CHINGMU_0800_follow}
WORKERS=${WORKERS:-25}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-0}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
MAX_LABELS=${MAX_LABELS:-}

ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_CHINGMU.json}

# Two separate dataset roots (preserve <dataset>/<scene>/...)
CAMERA_DATASET_DIR=${CAMERA_DATASET_DIR:-${OUTPUT_DIR}_camera}
ACTIONS_DATASET_DIR=${ACTIONS_DATASET_DIR:-${OUTPUT_DIR}_actions}

generate_assignment_manifest() {
  CONDA_ENV="${CONDA_ENV}" \
  ACTOR_ROOT="${ACTOR_ROOT}" \
  BAN_LIST="${BAN_LIST}" \
  ASSIGNMENTS_OUT="${ASSIGNMENTS_OUT}" \
  SCENES_DIR="${SCENES_DIR}" \
  TASKS_DIR="${TASKS_DIR}" \
  SEED="${SEED}" \
  EXCLUDE_DETAILED_LABELS="${EXCLUDE_DETAILED_LABELS}" \
  bash "${SCRIPT_DIR}/scripts/generate_assignment_manifest.sh"
}

if [ -n "${ASSIGNMENTS_OUT}" ] && [ ! -f "${ASSIGNMENTS_OUT}" ]; then
  echo "[RUN] Assignment manifest missing at ${ASSIGNMENTS_OUT}; generating..." >&2
  generate_assignment_manifest
fi

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/gen_path_dataset_telesim.py"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --camera-root "${CAMERA_DATASET_DIR}"
  --actions-root "${ACTIONS_DATASET_DIR}"
  --height-offset "${HEIGHT_OFFSET}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --scene-workers "${WORKERS}"
  --follow-distance 1.5
  --overwrite
)

if [ -n "${ASSIGNMENTS_OUT}" ] && [ -f "${ASSIGNMENTS_OUT}" ]; then
  cmd+=(--assignment-manifest "${ASSIGNMENTS_OUT}")
fi

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
if [ -n "${ASSIGNMENTS_OUT}" ] && [ -f "${ASSIGNMENTS_OUT}" ]; then
  echo "[GEN] assignment_manifest=${ASSIGNMENTS_OUT}" >&2
fi

if [ -n "${TAR_OUT:-}" ]; then
  cmd+=(--tar-out "${TAR_OUT}")
fi

"${cmd[@]}"
