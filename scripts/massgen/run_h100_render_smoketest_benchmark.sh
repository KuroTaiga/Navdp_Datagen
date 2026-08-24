#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-navdp-h100}"
USE_CONDA_RUN="${USE_CONDA_RUN:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-/private/dongjk/test_workflow}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${WORKFLOW_ROOT}/package}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKFLOW_ROOT}/results}"
RENDER_SCRIPT="${RENDER_SCRIPT:-${REPO_ROOT}/render_label_paths_telesim.py}"

VIDEO_BACKEND="${VIDEO_BACKEND:-nvenc}"
WORKERS="${WORKERS:-4}"
MINIMAL_FRAMES="${MINIMAL_FRAMES:-20}"
MAX_RENDERS="${MAX_RENDERS:-8}"
RENDERS_PER_FAMILY_SOURCE_SCENE="${RENDERS_PER_FAMILY_SOURCE_SCENE:-0}"
GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-0.2}"
CLEAN="${CLEAN:-false}"

# Set PATCH_HUMANS=true and ACTOR_SOURCE_IDS="7611 1018 ..." to repoint a
# copied smoke package to the avatars available on the H100 platform host.
PATCH_HUMANS="${PATCH_HUMANS:-false}"
HUMAN_ROOT="${HUMAN_ROOT:-/private/dongjk/navdata/human_gs_source}"
ACTOR_SOURCE_IDS="${ACTOR_SOURCE_IDS:-}"

if [ ! -f "${PACKAGE_ROOT}/smoketest_package_index.json" ]; then
  cat >&2 <<EOF
[ERROR] Missing package index: ${PACKAGE_ROOT}/smoketest_package_index.json

Copy or build the smoke package first. Expected platform layout:
  package: ${PACKAGE_ROOT}
  results: ${RESULTS_ROOT}
  scene data referenced by manifests: /team/telenav/navsources
  newer human avatars default: ${HUMAN_ROOT}
  older human avatars can be selected with HUMAN_ROOT=/team/telenav/human_avatars
EOF
  exit 2
fi

cmd_prefix=()
if [ "${USE_CONDA_RUN}" = "auto" ]; then
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
    USE_CONDA_RUN=false
  elif command -v conda >/dev/null 2>&1; then
    USE_CONDA_RUN=true
  else
    USE_CONDA_RUN=false
  fi
fi
if [ "${USE_CONDA_RUN}" = "true" ]; then
  cmd_prefix=(conda run --no-capture-output -n "${CONDA_ENV}")
  PYTHON_CMD=(python)
else
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN=python
    else
      echo "[ERROR] ${PYTHON_BIN} is not available and python was not found." >&2
      exit 1
    fi
  fi
  PYTHON_CMD=("${PYTHON_BIN}")
fi

if [ "${PATCH_HUMANS}" = "true" ]; then
  if [ -z "${ACTOR_SOURCE_IDS}" ]; then
    echo "[ERROR] PATCH_HUMANS=true requires ACTOR_SOURCE_IDS, for example: ACTOR_SOURCE_IDS=\"7611 1018\"." >&2
    exit 2
  fi
  patch_cmd=("${cmd_prefix[@]}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/patch_smoketest_package_humans.py")
  patch_cmd+=(--package-root "${PACKAGE_ROOT}")
  patch_cmd+=(--remote-human-root "${HUMAN_ROOT}")
  for actor_id in ${ACTOR_SOURCE_IDS}; do
    patch_cmd+=(--actor-source-id "${actor_id}")
  done
  echo "[H100] Patching human avatar paths under ${PACKAGE_ROOT}" >&2
  "${patch_cmd[@]}"
fi

run_cmd=("${cmd_prefix[@]}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/run_render_smoketest_benchmark.py")
run_cmd+=(--package-root "${PACKAGE_ROOT}")
run_cmd+=(--results-root "${RESULTS_ROOT}")
run_cmd+=(--repo-root "${REPO_ROOT}")
run_cmd+=(--render-script "${RENDER_SCRIPT}")
run_cmd+=(--video-backend "${VIDEO_BACKEND}")
run_cmd+=(--device cuda)
run_cmd+=(--workers "${WORKERS}")
run_cmd+=(--minimal-frames "${MINIMAL_FRAMES}")
run_cmd+=(--max-renders "${MAX_RENDERS}")
run_cmd+=(--renders-per-family-source-scene "${RENDERS_PER_FAMILY_SOURCE_SCENE}")
run_cmd+=(--gpu-sample-interval-sec "${GPU_SAMPLE_INTERVAL_SEC}")
if [ "${CLEAN}" = "true" ]; then
  run_cmd+=(--clean)
fi

echo "[H100] package_root=${PACKAGE_ROOT}" >&2
echo "[H100] results_root=${RESULTS_ROOT}" >&2
echo "[H100] workers=${WORKERS} max_renders=${MAX_RENDERS} minimal_frames=${MINIMAL_FRAMES} video_backend=${VIDEO_BACKEND}" >&2

exec "${run_cmd[@]}"
