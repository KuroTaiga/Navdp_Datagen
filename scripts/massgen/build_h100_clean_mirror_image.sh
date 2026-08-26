#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-navdp-datagen-h100:massgen}"
CUDA_BASE="${CUDA_BASE:-nvidia/cuda:12.1.1-devel-ubuntu22.04}"
CONDA_ENV="${CONDA_ENV:-cuda121}"
INSTALL_RENDER_REQUIREMENTS="${INSTALL_RENDER_REQUIREMENTS:-${INSTALL_LOCAL_EXTENSIONS:-1}}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
NO_CACHE="${NO_CACHE:-1}"
PULL="${PULL:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
KEEP_BUILD_CONTEXT="${KEEP_BUILD_CONTEXT:-0}"
SAVE_IMAGE_TAR="${SAVE_IMAGE_TAR:-}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_CONTEXT_PARENT="${BUILD_CONTEXT_PARENT:-${TMPDIR:-/tmp}}"
BUILD_CONTEXT_DIR=""

die() {
  echo "error: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_clean_git_tree() {
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    return 0
  fi

  local status
  status="$(git -C "${REPO_ROOT}" status --ignore-submodules=all --short --untracked-files=all)"
  if [[ -n "${status}" ]]; then
    echo "${status}" >&2
    die "working tree is not clean; commit/stash changes or set ALLOW_DIRTY=1 for a non-release test image"
  fi
}

check_required_sources() {
  local required=(
    environment.yml
    containers/h100/Dockerfile
    containers/h100/entrypoint.sh
    release/navdp_path_renderer/requirements.txt
    render_label_paths_telesim.py
    release/navdp_path_renderer/TeleSim3D/tele_sim/__init__.py
  )

  local path
  for path in "${required[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
      die "missing required source: ${path}"
    fi
  done
}

write_build_info() {
  local context_dir="$1"
  local info_path="${context_dir}/containers/h100/clean_image_build_info.json"
  local commit branch built_at
  commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  branch="$(git -C "${REPO_ROOT}" branch --show-current || true)"
  built_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  cat >"${info_path}" <<EOF
{
  "schema_version": "navdp_h100_clean_image_build.v1",
  "git_commit": "${commit}",
  "git_branch": "${branch}",
  "docker_platform": "${DOCKER_PLATFORM}",
  "cuda_base": "${CUDA_BASE}",
  "conda_env": "${CONDA_ENV}",
  "install_render_requirements": "${INSTALL_RENDER_REQUIREMENTS}",
  "built_at_utc": "${built_at}"
}
EOF
}

export_clean_context() {
  local context_dir="$1"
  (
    cd "${REPO_ROOT}"
    git ls-files -s -z \
      | while IFS= read -r -d '' entry; do
          local mode path
          mode="${entry%% *}"
          path="${entry#*$'\t'}"
          if [[ "${mode}" == "160000" ]]; then
            continue
          fi
          printf '%s\0' "${path}"
        done \
      | tar --no-recursion --null -T - -cf -
  ) | tar -C "${context_dir}" -xf -
}

cleanup() {
  if [[ -z "${BUILD_CONTEXT_DIR}" ]]; then
    return 0
  fi
  if [[ "${KEEP_BUILD_CONTEXT}" == "1" ]]; then
    echo "[clean-image] kept build context: ${BUILD_CONTEXT_DIR}"
  else
    rm -rf "${BUILD_CONTEXT_DIR}"
  fi
}

main() {
  require_command git
  require_command tar
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_command "${DOCKER_BIN}"
  fi

  check_clean_git_tree
  check_required_sources

  mkdir -p "${BUILD_CONTEXT_PARENT}"
  BUILD_CONTEXT_DIR="$(mktemp -d "${BUILD_CONTEXT_PARENT%/}/navdp_h100_clean_context.XXXXXX")"
  trap cleanup EXIT

  echo "[clean-image] exporting tracked superproject files to ${BUILD_CONTEXT_DIR}"
  export_clean_context "${BUILD_CONTEXT_DIR}"
  write_build_info "${BUILD_CONTEXT_DIR}"

  local docker_args=(
    build
    --platform "${DOCKER_PLATFORM}"
    -f containers/h100/Dockerfile
    --build-arg "CUDA_BASE=${CUDA_BASE}"
    --build-arg "CONDA_ENV=${CONDA_ENV}"
    --build-arg "INSTALL_RENDER_REQUIREMENTS=${INSTALL_RENDER_REQUIREMENTS}"
    -t "${IMAGE_TAG}"
  )
  if [[ "${PULL}" == "1" ]]; then
    docker_args+=(--pull)
  fi
  if [[ "${NO_CACHE}" == "1" ]]; then
    docker_args+=(--no-cache)
  fi

  echo "[clean-image] ${DOCKER_BIN} ${docker_args[*]} ${BUILD_CONTEXT_DIR}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[clean-image] dry run complete"
    return 0
  fi

  "${DOCKER_BIN}" "${docker_args[@]}" "${BUILD_CONTEXT_DIR}"

  if [[ -n "${SAVE_IMAGE_TAR}" ]]; then
    mkdir -p "$(dirname "${SAVE_IMAGE_TAR}")"
    echo "[clean-image] saving ${IMAGE_TAG} -> ${SAVE_IMAGE_TAR}"
    "${DOCKER_BIN}" save -o "${SAVE_IMAGE_TAR}" "${IMAGE_TAG}"
  fi
}

main "$@"
