#!/usr/bin/env bash
set -Eeuo pipefail

MIRROR_ROOT="${MIRROR_ROOT:-/team/telenav/code/Navdp_Datagen}"
REPO_URL="${REPO_URL:-https://github.com/KuroTaiga/Navdp_Datagen.git}"
BRANCH="${BRANCH:-massgen}"
PACKAGE_SRC="${PACKAGE_SRC:-}"
PACKAGE_DST="${PACKAGE_DST:-}"
BUILD_CONTAINER="${BUILD_CONTAINER:-0}"
CLEAN_IMAGE="${CLEAN_IMAGE:-1}"
UPDATE_SUBMODULES="${UPDATE_SUBMODULES:-0}"
IMAGE_TAG="${IMAGE_TAG:-navdp-datagen-h100:massgen}"

if [[ -d "${MIRROR_ROOT}/.git" ]]; then
  echo "[mirror] updating ${MIRROR_ROOT}"
  git -C "${MIRROR_ROOT}" fetch origin "${BRANCH}"
  git -C "${MIRROR_ROOT}" switch "${BRANCH}" || git -C "${MIRROR_ROOT}" switch -c "${BRANCH}" "origin/${BRANCH}"
  git -C "${MIRROR_ROOT}" pull --ff-only origin "${BRANCH}"
else
  echo "[mirror] cloning ${REPO_URL} -> ${MIRROR_ROOT}"
  mkdir -p "$(dirname "${MIRROR_ROOT}")"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${MIRROR_ROOT}"
fi

if [[ "${UPDATE_SUBMODULES}" == "1" ]]; then
  if ! git -C "${MIRROR_ROOT}" submodule update --init --recursive; then
    echo "[warn] submodule update failed" >&2
  fi
else
  echo "[mirror] skipping submodule init; H100 gsplat image installs render deps from pip"
fi

if [[ -n "${PACKAGE_SRC}" ]]; then
  if [[ -z "${PACKAGE_DST}" ]]; then
    echo "PACKAGE_DST must be set when PACKAGE_SRC is set" >&2
    exit 2
  fi
  mkdir -p "${PACKAGE_DST}"
  echo "[package] rsync ${PACKAGE_SRC}/ -> ${PACKAGE_DST}/"
  rsync -a --delete "${PACKAGE_SRC%/}/" "${PACKAGE_DST%/}/"
fi

echo "[preflight] repo status"
git -C "${MIRROR_ROOT}" status --short --branch

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "[warn] nvidia-smi not found"
fi

if [[ "${BUILD_CONTAINER}" == "1" ]]; then
  echo "[container] building ${IMAGE_TAG}"
  if [[ "${CLEAN_IMAGE}" == "1" ]]; then
    IMAGE_TAG="${IMAGE_TAG}" "${MIRROR_ROOT}/scripts/massgen/build_h100_clean_mirror_image.sh"
  else
    IMAGE_TAG="${IMAGE_TAG}" "${MIRROR_ROOT}/scripts/massgen/build_h100_container.sh"
  fi
fi

cat <<EOF

Mirror ready:
  MIRROR_ROOT=${MIRROR_ROOT}
  BRANCH=${BRANCH}

EOF

run_header="Next:"
if [[ "${BUILD_CONTAINER}" != "1" ]]; then
  run_header="Then:"
  cat <<EOF
Next:
  cd ${MIRROR_ROOT}
  IMAGE_TAG=${IMAGE_TAG} scripts/massgen/build_h100_clean_mirror_image.sh

EOF
fi

cat <<EOF
${run_header}
  cd ${MIRROR_ROOT}
  PACKAGE_ROOT=<package path> RESULTS_ROOT=<output path> \\
    scripts/massgen/run_h100_container.sh run

Or without container:
  /path/to/cuda-env/bin/python scripts/massgen/run_family_rollout_h100.py \\
    --package-root <package path> \\
    --results-root <output path> \\
    --python-bin /path/to/cuda-env/bin/python \\
    --gpu-devices 0,1,2,3 --cpu-cores 120 --jobs-per-gpu 4 \\
    --video-backend cpu --renders-per-family-source-scene 50 \\
    --minimal-frames 16 --command-attempts 3 --clean
EOF
