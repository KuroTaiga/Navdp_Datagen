#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-navdp-datagen-h100:massgen}"
CUDA_BASE="${CUDA_BASE:-nvidia/cuda:12.1.1-devel-ubuntu22.04}"
CONDA_ENV="${CONDA_ENV:-cuda121}"
INSTALL_RENDER_REQUIREMENTS="${INSTALL_RENDER_REQUIREMENTS:-${INSTALL_LOCAL_EXTENSIONS:-1}}"

cd "${REPO_ROOT}"

docker build \
  -f containers/h100/Dockerfile \
  --build-arg CUDA_BASE="${CUDA_BASE}" \
  --build-arg CONDA_ENV="${CONDA_ENV}" \
  --build-arg INSTALL_RENDER_REQUIREMENTS="${INSTALL_RENDER_REQUIREMENTS}" \
  -t "${IMAGE_TAG}" \
  .
