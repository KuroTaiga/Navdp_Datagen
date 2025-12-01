#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR" > /dev/null

RENDERS_DIR=${RENDERS_DIR-/mnt/nas/jiankundong/random_human_dataset_w_ban_65k}
REPORT_DIR=${REPORT_DIR:-./data/post_datagen_analysis_65k}
TASKS_DIR=${TASKS_DIR-./data/selected_65k}
ASSIGNMENTS_JSON=${ASSIGNMENTS_JSON:-./data/actor_assignments_w_ban_65k.json}
VIDEO_MIN_MB=${VIDEO_MIN_MB:-1}
TOP_SCENES=${TOP_SCENES:-20}
HIST_BINS=${HIST_BINS:-60}

if [ ! -d "${RENDERS_DIR}" ]; then
  echo "ERROR: Renders directory not found: ${RENDERS_DIR}" >&2
  exit 1
fi

if [ -d "${REPORT_DIR}" ]; then
  rm -rf "${REPORT_DIR}"
fi
mkdir -p "${REPORT_DIR}"

CMD=(
  conda run --no-capture-output -n cuda121 python post_datagen_analysis.py
  --renders-dir "${RENDERS_DIR}"
  --report-dir "${REPORT_DIR}"
  --top-scenes "${TOP_SCENES}"
  --hist-bins "${HIST_BINS}"
  --video-min-mb "${VIDEO_MIN_MB}"
)

if [ -n "${TASKS_DIR}" ]; then
  if [ ! -d "${TASKS_DIR}" ]; then
    echo "ERROR: Tasks directory not found: ${TASKS_DIR}" >&2
    exit 1
  fi
  CMD+=(--tasks-dir "${TASKS_DIR}")
fi

if [ -n "${ASSIGNMENTS_JSON}" ]; then
  if [ ! -f "${ASSIGNMENTS_JSON}" ]; then
    echo "ERROR: Assignments JSON not found: ${ASSIGNMENTS_JSON}" >&2
    exit 1
  fi
  CMD+=(--assignments-json "${ASSIGNMENTS_JSON}")
fi

echo "Analyzing rendered outputs under ${RENDERS_DIR} ..."
"${CMD[@]}" "$@"

popd > /dev/null
