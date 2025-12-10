#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR" > /dev/null

ASSIGNMENTS_JSON=${ASSIGNMENTS_JSON:-./data/actor_assignments_w_ban_33w.json}
OUTPUT_DIR=${OUTPUT_DIR:-./data/actor_assignment_analysis_33w}
TASKS_DIR=${TASKS_DIR:-./data/selected_33w}
TOP_ACTORS=${TOP_ACTORS:-30}
HIST_BINS=${HIST_BINS:-60}

if [ ! -f "${ASSIGNMENTS_JSON}" ]; then
  echo "ERROR: Assignments JSON not found: ${ASSIGNMENTS_JSON}" >&2
  exit 1
fi

if [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

CMD=(
  conda run --no-capture-output -n cuda121 python actor_assignment_analysis.py
  --assignments-json "${ASSIGNMENTS_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --hist-bins "${HIST_BINS}"
  --top-actors "${TOP_ACTORS}"
)

if [ -n "${TASKS_DIR}" ]; then
  if [ ! -d "${TASKS_DIR}" ]; then
    echo "ERROR: Tasks directory not found: ${TASKS_DIR}" >&2
    exit 1
  fi
  CMD+=(--tasks-dir "${TASKS_DIR}")
fi

echo "Analyzing actor assignments from ${ASSIGNMENTS_JSON} ..."
"${CMD[@]}" "$@"

popd > /dev/null
