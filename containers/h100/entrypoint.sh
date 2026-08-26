#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Navdp_Datagen}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/${CONDA_ENV:-cuda121}/bin/python}"
PACKAGE_ROOT="${PACKAGE_ROOT:-}"
RESULTS_ROOT="${RESULTS_ROOT:-/work/h100_results}"
GPU_DEVICES="${GPU_DEVICES:-0}"
CPU_CORES="${CPU_CORES:-$(nproc)}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-cpu}"
COMMAND_ATTEMPTS="${COMMAND_ATTEMPTS:-3}"
RENDERS_PER_FAMILY_SOURCE_SCENE="${RENDERS_PER_FAMILY_SOURCE_SCENE:-50}"
MINIMAL_FRAMES="${MINIMAL_FRAMES:-16}"
EXTRA_H100_ARGS="${EXTRA_H100_ARGS:-}"

cd "${REPO_ROOT}"

preflight() {
  local failed=0
  mkdir -p "${RESULTS_ROOT}" "${MPLCONFIGDIR:-/tmp/navdp_mplconfig}" "${XDG_CACHE_HOME:-/tmp/navdp_xdg_cache}" "${FC_CACHEDIR:-/tmp/navdp_fontconfig}"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available in the container" >&2
    failed=1
  else
    nvidia-smi || failed=1
  fi

  if [[ -n "${PACKAGE_ROOT}" && ! -d "${PACKAGE_ROOT}" ]]; then
    echo "PACKAGE_ROOT does not exist: ${PACKAGE_ROOT}" >&2
    failed=1
  fi

  if [[ ! -w "${RESULTS_ROOT}" ]]; then
    echo "RESULTS_ROOT is not writable: ${RESULTS_ROOT}" >&2
    failed=1
  fi

  "${PYTHON_BIN}" - <<'PY' || failed=1
from pathlib import Path
import sys
import torch

repo = Path.cwd()
sys.path.insert(0, str(repo))
telesim_root = repo / "TeleSim3D"
if not (telesim_root / "tele_sim").exists():
    telesim_root = repo / "release" / "navdp_path_renderer" / "TeleSim3D"
sys.path.insert(0, str(telesim_root))

import imageio  # noqa: F401
import numpy  # noqa: F401
import gsplat  # noqa: F401
import tele_sim  # noqa: F401

print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("gsplat", getattr(gsplat, "__version__", "unknown"))
print("tele_sim", tele_sim.__file__)
PY

  return "${failed}"
}

run_h100() {
  if [[ -z "${PACKAGE_ROOT}" ]]; then
    echo "PACKAGE_ROOT must be set for run" >&2
    return 2
  fi

  local cmd=(
    "${PYTHON_BIN}" scripts/massgen/run_family_rollout_h100.py
    --package-root "${PACKAGE_ROOT}"
    --results-root "${RESULTS_ROOT}"
    --python-bin "${PYTHON_BIN}"
    --repo-root "${REPO_ROOT}"
    --gpu-devices "${GPU_DEVICES}"
    --cpu-cores "${CPU_CORES}"
    --jobs-per-gpu "${JOBS_PER_GPU}"
    --video-backend "${VIDEO_BACKEND}"
    --command-attempts "${COMMAND_ATTEMPTS}"
  )

  if [[ -n "${RENDERS_PER_FAMILY_SOURCE_SCENE}" && "${RENDERS_PER_FAMILY_SOURCE_SCENE}" != "0" ]]; then
    cmd+=(--renders-per-family-source-scene "${RENDERS_PER_FAMILY_SOURCE_SCENE}")
  fi
  if [[ -n "${MINIMAL_FRAMES}" && "${MINIMAL_FRAMES}" != "0" ]]; then
    cmd+=(--minimal-frames "${MINIMAL_FRAMES}")
  fi
  if [[ -n "${EXTRA_H100_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( ${EXTRA_H100_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  exec "${cmd[@]}"
}

case "${1:-preflight}" in
  preflight)
    preflight
    ;;
  run)
    preflight
    run_h100
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
