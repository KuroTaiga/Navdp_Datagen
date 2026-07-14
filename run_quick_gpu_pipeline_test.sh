#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV=${CONDA_ENV:-cuda121}
export PYTHONUNBUFFERED=1
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/data/quick_gpu_pipeline_test"}
export GOLDEN_ROOT=${GOLDEN_ROOT:-"${SCRIPT_DIR}/data2/0500_fpv"}
export COMPARE_OUT_JSON=${COMPARE_OUT_JSON:-"${SCRIPT_DIR}/analysis/quick_gpu_camera_compare.json"}
export TASKS_DIR=${TASKS_DIR:-"${SCRIPT_DIR}/data/interiorGS_0500_42"}
export SCENES_DIR=${SCENES_DIR:-"${SCRIPT_DIR}/data/scenes"}
SCENE_ID=${SCENE_ID:-}
if [ -n "$SCENE_ID" ]; then
  export SCENE_ID
fi

for arg in "$@"; do
  case "$arg" in
    --output-root=*)
      export OUTPUT_ROOT="${arg#*=}"
      ;;
  esac
done
for ((i=1; i<=$#; i++)); do
  if [ "${!i}" = "--output-root" ]; then
    next_index=$((i + 1))
    if [ $next_index -le $# ]; then
      export OUTPUT_ROOT="${!next_index}"
    fi
  fi
done

conda run --no-capture-output -n "$CONDA_ENV" python "${SCRIPT_DIR}/scripts/quick_gpu_pipeline_test.py" \
  --render-script "${SCRIPT_DIR}/render_label_paths_telesim.py" \
  "$@"

# Camera comparison runs inside quick_gpu_pipeline_test.py by default.
