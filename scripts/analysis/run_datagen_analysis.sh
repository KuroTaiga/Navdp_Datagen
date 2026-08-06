#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
pushd "$REPO_ROOT" > /dev/null

TASKS_DIR=${TASKS_DIR:-./data/selected_33w}
OUTPUT_DIR=${OUTPUT_DIR:-./data/datagen_analysis_33w}
TOP_SCENES=${TOP_SCENES:-20}
HIST_BINS=${HIST_BINS:-60}

if [ ! -d "${TASKS_DIR}" ]; then
  echo "ERROR: Tasks directory not found: ${TASKS_DIR}" >&2
  exit 1
fi

if [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Analyzing data generation tasks under ${TASKS_DIR} ..."
conda run --no-capture-output -n cuda121 python "${SCRIPT_DIR}/datagen_analysis.py" \
  --tasks-dir "${TASKS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-scenes "${TOP_SCENES}" \
  --hist-bins "${HIST_BINS}" \
  "$@"

popd > /dev/null
