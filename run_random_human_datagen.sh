#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

# Storage toggles (set env vars to true/false/yes/no)
ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE:-true}
ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE:-false}
ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE:-true}
CLEAR_LOCAL_OUTPUT_DIR=${CLEAR_LOCAL_OUTPUT_DIR:-true}

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
  exit 1
fi

SEED=${SEED:-12345}
CONDA_ENV=${CONDA_ENV:-cuda121}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_65k.json}
SCENES_DIR=${SCENES_DIR:-./data/scenes}
TASKS_DIR=${TASKS_DIR:-./data/selected_65k}
OUTPUT_DIR=${OUTPUT_DIR:-./data/path_video_frames_random_humans_65k}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/random_human_dataset_w_ban_65k}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/srv/navdp}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-user@other-training-pc}
LOCAL_OUTPUT_BASENAME="$(basename "$OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"
WORKERS=${WORKERS:-3}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-38}

ensure_writable_dir() {
  local target="$1"
  if [ ! -d "$target" ]; then
    mkdir -p "$target"
  fi
  if [ ! -w "$target" ]; then
    chmod 777 "$target"
  fi
  if [ ! -w "$target" ]; then
    echo "ERROR: Output directory $target is not writable." >&2
    exit 1
  fi
}

prepare_local_output_dir() {
  local target="$1"
  ensure_writable_dir "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

prepare_local_output_dir "$OUTPUT_DIR"

if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  NAS_TEST_DIR="${OFFLOAD_NAS_DIR}/__connectivity_check__"
  if mkdir -p "${NAS_TEST_DIR}" \
    && : > "${NAS_TEST_DIR}/.touch" \
    && rm -f "${NAS_TEST_DIR}/.touch"; then
    echo "[CHECK] NAS reachable at ${OFFLOAD_NAS_DIR}"
  else
    echo "[CHECK] ERROR: cannot write to NAS ${OFFLOAD_NAS_DIR}" >&2
    exit 1
  fi
fi

echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"

echo "[RUN] Random human data generation, seed ${SEED}"
conda run --no-capture-output -n "$CONDA_ENV" python random_actor_assignments.py \
  --actor-root "${ACTOR_ROOT}" \
  --ban-list "${BAN_LIST}" \
  --assignments-out "${ASSIGNMENTS_OUT}" \
  --scenes-dir "${SCENES_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --seed "${SEED}"
echo "Assignment manifest generated at ${ASSIGNMENTS_OUT}"

render_extra_snippets=(
  "--overwrite --stabilize --gpu-only --show-BEV --no-rgb-frames --navdp-ply-per-scene"
)
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  render_extra_snippets+=("--offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb ${OFFLOAD_MIN_FREE_GB}")
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" python parallel_render_paths.py
  --assignment-manifest "${ASSIGNMENTS_OUT}"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --output-dir "${OUTPUT_DIR}"
)
for snippet in "${render_extra_snippets[@]}"; do
  parallel_cmd+=(--render-extra-args "$snippet")
done

render_status=0
set +e
"${parallel_cmd[@]}"
render_status=$?
set -e
if [ $render_status -ne 0 ]; then
  echo "[WARN] parallel_render_paths.py exited with status ${render_status}, continuing per request."
fi

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  storage_sync_remote "$OUTPUT_DIR" "$REMOTE_TARGET_DIR"
fi

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE"; then
  if [ -d "$OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
  fi
fi

exit $render_status
