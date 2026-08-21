#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-navdp-datagen-h100:massgen}"
PACKAGE_ROOT="${PACKAGE_ROOT:?Set PACKAGE_ROOT to the MassGen render package path on the host}"
RESULTS_ROOT="${RESULTS_ROOT:?Set RESULTS_ROOT to the desired output path on the host}"
GPU_DEVICES="${GPU_DEVICES:-0}"
CPU_CORES="${CPU_CORES:-120}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
MINIMAL_FRAMES="${MINIMAL_FRAMES:-16}"
RENDERS_PER_FAMILY_SOURCE_SCENE="${RENDERS_PER_FAMILY_SOURCE_SCENE:-50}"
COMMAND_ATTEMPTS="${COMMAND_ATTEMPTS:-3}"
EXTRA_H100_ARGS="${EXTRA_H100_ARGS:-}"
CONTAINER_REPO_ROOT="${CONTAINER_REPO_ROOT:-/workspace/Navdp_Datagen}"

declare -a docker_args=(
  run
  --rm
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e "PACKAGE_ROOT=${PACKAGE_ROOT}"
  -e "RESULTS_ROOT=${RESULTS_ROOT}"
  -e "GPU_DEVICES=${GPU_DEVICES}"
  -e "CPU_CORES=${CPU_CORES}"
  -e "JOBS_PER_GPU=${JOBS_PER_GPU}"
  -e "MINIMAL_FRAMES=${MINIMAL_FRAMES}"
  -e "RENDERS_PER_FAMILY_SOURCE_SCENE=${RENDERS_PER_FAMILY_SOURCE_SCENE}"
  -e "COMMAND_ATTEMPTS=${COMMAND_ATTEMPTS}"
  -e "EXTRA_H100_ARGS=${EXTRA_H100_ARGS}"
  -e "REPO_ROOT=${CONTAINER_REPO_ROOT}"
)

mount_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    docker_args+=(-v "${path}:${path}")
  fi
}

mount_if_exists "${PACKAGE_ROOT}"
mkdir -p "${RESULTS_ROOT}"
mount_if_exists "${RESULTS_ROOT}"
mount_if_exists /mnt/DATA
mount_if_exists /mnt/DATA1
mount_if_exists /private_lxh
mount_if_exists /team/telenav

if [[ "${MOUNT_REPO:-1}" == "1" ]]; then
  docker_args+=(-v "${REPO_ROOT}:${CONTAINER_REPO_ROOT}")
fi

docker "${docker_args[@]}" "${IMAGE_TAG}" "${1:-run}"
