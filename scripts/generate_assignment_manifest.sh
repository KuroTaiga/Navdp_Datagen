#!/usr/bin/env bash
set -euo pipefail

# Generate a random actor->label assignment manifest for the "following" pipeline.
# This is shared by both the CUDA pipeline and the TeleSim pipeline runners.
#
# Inputs are taken from environment variables to keep call-sites simple:
# - CONDA_ENV (default: cuda121)
# - ACTOR_ROOT (required)
# - BAN_LIST (optional)
# - ASSIGNMENTS_OUT (required)
# - SCENES_DIR (required)
# - TASKS_DIR (required)
# - SEED (optional; default: 1)
# - EXCLUDE_DETAILED_LABELS (true/false; default: true)

CONDA_ENV=${CONDA_ENV:-cuda121}
SEED=${SEED:-1}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}

ACTOR_ROOT=${ACTOR_ROOT:-}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-}
SCENES_DIR=${SCENES_DIR:-}
TASKS_DIR=${TASKS_DIR:-}
BAN_LIST=${BAN_LIST:-}

if [ -z "${ACTOR_ROOT}" ]; then
  echo "[ASSIGN] ERROR: ACTOR_ROOT is required." >&2
  exit 1
fi
if [ -z "${ASSIGNMENTS_OUT}" ]; then
  echo "[ASSIGN] ERROR: ASSIGNMENTS_OUT is required." >&2
  exit 1
fi
if [ -z "${SCENES_DIR}" ]; then
  echo "[ASSIGN] ERROR: SCENES_DIR is required." >&2
  exit 1
fi
if [ -z "${TASKS_DIR}" ]; then
  echo "[ASSIGN] ERROR: TASKS_DIR is required." >&2
  exit 1
fi

mkdir -p "$(dirname "${ASSIGNMENTS_OUT}")"

detailed_flag="--exclude-detailed-labels"
if [ "${EXCLUDE_DETAILED_LABELS}" = "false" ]; then
  detailed_flag="--no-exclude-detailed-labels"
fi

cmd=(conda run --no-capture-output -n "${CONDA_ENV}" python random_actor_assignments.py)
cmd+=(--actor-root "${ACTOR_ROOT}")
cmd+=(--assignments-out "${ASSIGNMENTS_OUT}")
cmd+=(--scenes-dir "${SCENES_DIR}")
cmd+=(--tasks-dir "${TASKS_DIR}")
cmd+=(--seed "${SEED}")
cmd+=("${detailed_flag}")

if [ -n "${BAN_LIST}" ]; then
  cmd+=(--ban-list "${BAN_LIST}")
fi

echo "[ASSIGN] Generating assignment manifest (seed=${SEED})" >&2
echo "[ASSIGN] actor_root=${ACTOR_ROOT}" >&2
echo "[ASSIGN] tasks_dir=${TASKS_DIR}" >&2
echo "[ASSIGN] scenes_dir=${SCENES_DIR}" >&2
echo "[ASSIGN] out=${ASSIGNMENTS_OUT}" >&2

"${cmd[@]}"

echo "[ASSIGN] Done: ${ASSIGNMENTS_OUT}" >&2

