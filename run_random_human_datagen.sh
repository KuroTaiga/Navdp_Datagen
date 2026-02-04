#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

# Ensure we have a Python interpreter available (needed for path resolution helper below).

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

# Tiny helper for consistent usage errors so RESUME mode is easy to discover.
show_usage_and_exit() {
  echo "Usage: $(basename "$0") [RESUME [LOG_PATH]] [--pipeline gpu|legacy] [--legacy-pipeline]" >&2
  exit 1
}

# CLI parsing: optional RESUME enables resume mode; optional log path accepted.
PIPELINE_MODE=${PIPELINE_MODE:-legacy}
RESUME_MODE=false
RESUME_LOG_PATH="CHINGMU_0800_follow_1.log"
while [ $# -gt 0 ]; do
  case "$1" in
    RESUME)
      RESUME_MODE=true
      shift
      if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
        RESUME_LOG_PATH="$1"
        shift
      fi
      ;;
    --pipeline)
      shift
      if [ $# -eq 0 ]; then
        echo "[ERROR] --pipeline expects a value (gpu|legacy)." >&2
        show_usage_and_exit
      fi
      PIPELINE_MODE="$1"
      shift
      ;;
    --legacy-pipeline)
      PIPELINE_MODE="legacy"
      shift
      ;;
    --gpu-pipeline)
      PIPELINE_MODE="gpu"
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      show_usage_and_exit
      ;;
  esac
done
PIPELINE_MODE=$(echo "$PIPELINE_MODE" | tr '[:upper:]' '[:lower:]')
if [ "$PIPELINE_MODE" != "gpu" ] && [ "$PIPELINE_MODE" != "legacy" ]; then
  echo "[ERROR] Invalid --pipeline value: ${PIPELINE_MODE} (expected gpu|legacy)." >&2
  show_usage_and_exit
fi

FFMPEG_BIN=${FFMPEG_BIN:-}
if [ -n "$FFMPEG_BIN" ]; then
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
elif [ "$PIPELINE_MODE" = "gpu" ] && command -v ffmpeg >/dev/null 2>&1; then
  export IMAGEIO_FFMPEG_EXE
  IMAGEIO_FFMPEG_EXE=$(command -v ffmpeg)
fi

# Convenience wrapper so we can expand relative paths and keep the script POSIX-ish.
abspath() {
  "$PYTHON_BIN" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# -----------------------------------------------------------------------------
# User-configurable defaults (override via env vars before invoking the script).
# Collect everything editable here so deployments only need to tweak this block.
# -----------------------------------------------------------------------------
# Storage toggles so the same runner can ship data to different targets.
ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE:-true}
ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE:-false}
ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE:-false}
CLEAR_LOCAL_OUTPUT_DIR=${CLEAR_LOCAL_OUTPUT_DIR:-false}

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
  exit 1
fi

REMOTE_ONLY_STORAGE=false
if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  REMOTE_ONLY_STORAGE=true
fi
: "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=false}"
if [ "$REMOTE_ONLY_STORAGE" = true ]; then
  : "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=true}"
fi
ENABLE_REMOTE_STORAGE_GUARD=${ENABLE_REMOTE_STORAGE_GUARD:-true}
REMOTE_STORAGE_LIMIT_GB=${REMOTE_STORAGE_LIMIT_GB:-500}
REMOTE_STORAGE_RESUME_GB=${REMOTE_STORAGE_RESUME_GB:-200}
REMOTE_STORAGE_GUARD_INTERVAL_SECS=${REMOTE_STORAGE_GUARD_INTERVAL_SECS:-60}
REMOTE_STORAGE_GUARD_ENABLED=false
if [ "$REMOTE_ONLY_STORAGE" = true ] && storage_bool_true "$ENABLE_REMOTE_STORAGE_GUARD"; then
  REMOTE_STORAGE_GUARD_ENABLED=true
fi
SETSID_BIN=""
if command -v setsid >/dev/null 2>&1; then
  SETSID_BIN=$(command -v setsid)
fi
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ] && [ -z "$SETSID_BIN" ]; then
  echo "[STORAGE] WARN: setsid is unavailable; disabling remote storage guard throttling." >&2
  REMOTE_STORAGE_GUARD_ENABLED=false
fi

# Core configuration for assignment planning + rendering. Most callers just tweak
# DATA roots or seeds via environment variables.
SEED=${SEED:-1}
CONDA_ENV=${CONDA_ENV:-cuda121}
# ACTOR_ROOT=${ACTOR_ROOT:-./data/SHHQ_gs/walking}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_CHINGMU.json}
PARALLEL_REPORT_DIR=${PARALLEL_REPORT_DIR:-./parallel_render_report_CHINGMU_key1.json}
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_scenes_rescaled}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./navdata/CHINGMU_0800_follow_flaw}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/random_human_dataset_w_ban_33w_1}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
PROGRESS_JSON=${PROGRESS_JSON:-./analysis/CHINGMU_0800_follow_1.json}
STATUS_JSON=${STATUS_JSON:-./analysis/CHINGMU_0800_follow_status.json}
PER_JOB_METRICS_DIR=${PER_JOB_METRICS_DIR:-./analysis/CHINGMU_0800_follow_metrics}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/mnt/DATA/navdp_data_33w_1}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-lenovo@192.168.151.40}
LOCAL_OUTPUT_BASENAME="$(basename "$OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"
REMOTE_SYNC_INTERVAL_SECS=${REMOTE_SYNC_INTERVAL_SECS:-120}
# Optional NPC placement/debug (applies to FPV or following data). Leave values empty to skip.
NPC_ENABLE=${NPC_ENABLE:-false}                            # true/false to append NPC args
NPC_DENSITY_COVERAGE=${NPC_DENSITY_COVERAGE:-0.3}          # e.g., 0.2 angular coverage
NPC_COUNT=${NPC_COUNT:-8}                                  # desired NPC count per frame
NPC_PRIORITY=${NPC_PRIORITY:-coverage}                     # coverage|count
NPC_MAX_RANGE=${NPC_MAX_RANGE:-10}                         # meters, radial cap
NPC_FREE_THRESHOLD=${NPC_FREE_THRESHOLD:-250}              # occupancy threshold
NPC_FREE_WHITE=${NPC_FREE_WHITE:-true}                     # true => free is white >= threshold
NPC_DENSITY_MODE=${NPC_DENSITY_MODE:-angular}              # angular|area
NPC_ZONE_RATIO=${NPC_ZONE_RATIO:-1:2:1}                    # near:mid:far ratio (applied when count>=12)
NPC_EXTRA_FLAGS=${NPC_EXTRA_FLAGS:-}                       # any extra passthrough (e.g., --npc-bev-debug)
NPC_FRAME_POOL_SIZE=${NPC_FRAME_POOL_SIZE:-50}             # preload this many NPC PLY frames per worker
NPC_PLACEMENT_BACKEND=${NPC_PLACEMENT_BACKEND:-}
WORKERS=${WORKERS:- 24}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-38}
# Robot camera stats
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}
# vram reserve function
RETRY_CUDA_OOM=${RETRY_CUDA_OOM:-true}
CUDA_OOM_RETRY_DELAY=${CUDA_OOM_RETRY_DELAY:-10}
CUDA_OOM_MAX_RETRIES=${CUDA_OOM_MAX_RETRIES:--1}
# set files types we want as output
# To enable per-path BEV debug images, run: ENABLE_BEV_IMAGES=true ./run_random_human_datagen.sh
ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-false}
ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-false}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-false}
ENABLE_CAMERA_METADATA=${ENABLE_CAMERA_METADATA:-true}
ENABLE_FOLLOW_METADATA=${ENABLE_FOLLOW_METADATA:-true}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}

GPU_ONLY_FLAG="--gpu-only"
if [ "$PIPELINE_MODE" = "legacy" ]; then
  : "${PLY_TRANSFORM_BACKEND:=cpu}"
  : "${VIDEO_BACKEND:=cpu}"
  : "${NPC_PLACEMENT_BACKEND:=cpu}"
  : "${STRICT_GPU_BACKENDS:=false}"
else
  : "${PLY_TRANSFORM_BACKEND:=gpu}"
  : "${VIDEO_BACKEND:=nvenc}"
  : "${NPC_PLACEMENT_BACKEND:=gpu}"
  : "${STRICT_GPU_BACKENDS:=true}"
fi
export STRICT_GPU_BACKENDS

# Default render_label_paths.py snippets appended to every worker invocation.
render_extra_args="--overwrite --stabilize ${GPU_ONLY_FLAG} --navdp-ply-per-scene --height-offset ${HEIGHT_OFFSET} --no-validate-path-bounds"
render_extra_args+=" --ply-transform-backend ${PLY_TRANSFORM_BACKEND}"
render_extra_args+=" --video-backend ${VIDEO_BACKEND}"
if storage_bool_true "$ENABLE_BEV_IMAGES"; then
  render_extra_args+=' --show-BEV'
else
  render_extra_args+=' --no-show-BEV'
fi
if storage_bool_true "$ENABLE_VIDEO_OUTPUT"; then
  render_extra_args+=' --video'
else
  render_extra_args+=' --no-video'
fi
if [ -n "${VIDEO_NVENC_PRESET:-}" ]; then
  render_extra_args+=" --video-nvenc-preset ${VIDEO_NVENC_PRESET}"
fi
if [ -n "${VIDEO_NVENC_BITRATE:-}" ]; then
  render_extra_args+=" --video-nvenc-bitrate ${VIDEO_NVENC_BITRATE}"
fi

# GPU video backend tuning for PyNvVideoCodec (avoid frame reordering/jitter).
if [ "${VIDEO_BACKEND}" = "gpu" ]; then
  : "${GPU_VIDEO_DISABLE_BFRAMES:=1}"
  : "${GPU_VIDEO_CLONE:=1}"
  : "${GPU_VIDEO_SYNC:=both}"
  : "${GPU_VIDEO_RETAIN_FRAMES:=4}"
  export GPU_VIDEO_DISABLE_BFRAMES GPU_VIDEO_CLONE GPU_VIDEO_SYNC GPU_VIDEO_RETAIN_FRAMES
  echo "[VIDEO] GPU backend: bframes=${GPU_VIDEO_DISABLE_BFRAMES} clone=${GPU_VIDEO_CLONE} sync=${GPU_VIDEO_SYNC} retain=${GPU_VIDEO_RETAIN_FRAMES}" >&2
fi
if storage_bool_true "$ENABLE_RGB_FRAMES"; then
  render_extra_args+=' --rgb-frames'
else
  render_extra_args+=' --no-rgb-frames'
fi
if storage_bool_true "$ENABLE_DEPTH_OUTPUT"; then
  render_extra_args+=' --save-depth-maps'
else
  render_extra_args+=' --no-save-depth-maps'
fi
if storage_bool_true "$ENABLE_CAMERA_METADATA"; then
  render_extra_args+=' --save-camera-metadata'
else
  render_extra_args+=' --no-save-camera-metadata'
fi
if storage_bool_true "$ENABLE_FOLLOW_METADATA"; then
  render_extra_args+=' --save-follow-metadata'
else
  render_extra_args+=' --no-save-follow-metadata'
fi
render_extra_snippets=("$render_extra_args")
# Optional NPC placement / BEV debug flags
if storage_bool_true "$NPC_ENABLE"; then
  npc_args=(
    "--npc-render"
    "--npc-density-mode ${NPC_DENSITY_MODE}"
    "--npc-priority ${NPC_PRIORITY}"
    "--npc-zone-ratio ${NPC_ZONE_RATIO}"
    "--npc-free-threshold ${NPC_FREE_THRESHOLD}"
  )
  if storage_bool_true "$NPC_FREE_WHITE"; then
    npc_args+=("--npc-free-white")
  else
    npc_args+=("--no-npc-free-white")
  fi
  if [ -n "${NPC_DENSITY_COVERAGE:-}" ]; then
    npc_args+=("--npc-density-coverage ${NPC_DENSITY_COVERAGE}")
  fi
  if [ -n "${NPC_COUNT:-}" ]; then
    npc_args+=("--npc-count ${NPC_COUNT}")
  fi
  if [ -n "${NPC_MAX_RANGE:-}" ]; then
    npc_args+=("--npc-max-range ${NPC_MAX_RANGE}")
  fi
  if [ -n "${NPC_FRAME_POOL_SIZE:-}" ]; then
    npc_args+=("--npc-frame-pool-size ${NPC_FRAME_POOL_SIZE}")
  fi
  if [ -n "${NPC_PLACEMENT_BACKEND:-}" ]; then
    npc_args+=("--npc-placement-backend ${NPC_PLACEMENT_BACKEND}")
  fi
  # Auto-clearance from SHHQ sources; keep on by default when NPC_ENABLE is true.
  npc_args+=("--npc-auto-clearance" "--npc-actor-root ${ACTOR_ROOT}")
  if [ -n "${NPC_EXTRA_FLAGS:-}" ]; then
    npc_args+=(${NPC_EXTRA_FLAGS})
  fi
  render_extra_snippets+=("${npc_args[*]}")
fi
if ! [[ "$REMOTE_STORAGE_LIMIT_GB" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_LIMIT_GB must be an integer value (received '$REMOTE_STORAGE_LIMIT_GB')." >&2
  exit 1
fi
if ! [[ "$REMOTE_STORAGE_RESUME_GB" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB must be an integer value (received '$REMOTE_STORAGE_RESUME_GB')." >&2
  exit 1
fi
if [ "$REMOTE_STORAGE_RESUME_GB" -ge "$REMOTE_STORAGE_LIMIT_GB" ]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB (${REMOTE_STORAGE_RESUME_GB}GB) must be less than REMOTE_STORAGE_LIMIT_GB (${REMOTE_STORAGE_LIMIT_GB}GB)." >&2
  exit 1
fi
if ! [[ "$REMOTE_STORAGE_GUARD_INTERVAL_SECS" =~ ^[0-9]+$ ]]; then
  echo "[STORAGE] ERROR: REMOTE_STORAGE_GUARD_INTERVAL_SECS must be an integer value (received '$REMOTE_STORAGE_GUARD_INTERVAL_SECS')." >&2
  exit 1
fi

REMOTE_SYNC_WORKER_PID=""
REMOTE_SYNC_DONE_FILE=""
REMOTE_STORAGE_UNAVAILABLE=false
PARALLEL_PID=""
PARALLEL_GROUP_ID=""
STOP_REQUESTED=false

handle_remote_storage_unavailable() {
  if [ "$REMOTE_STORAGE_UNAVAILABLE" = true ]; then
    return
  fi
  REMOTE_STORAGE_UNAVAILABLE=true
  echo "[STORAGE] Remote destination unavailable; pausing generation to avoid data loss." >&2
  if [ -n "$PARALLEL_GROUP_ID" ]; then
    kill -TERM -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
  elif [ -n "$PARALLEL_PID" ]; then
    kill -TERM "$PARALLEL_PID" >/dev/null 2>&1 || true
  fi
}

remote_sync_worker_loop() {
  local source_dir="$1"
  local remote_dir="$2"
  local done_flag="$3"
  local interval_secs="$4"
  local abort_on_failure="${5:-false}"
  local parent_pid="${6:-}"
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  echo "[STORAGE] Remote sync worker started for ${source_dir} -> ${REMOTE_SSH_TARGET:-?}:${remote_dir} (interval ${interval_secs}s)"
  local iteration=0
  while true; do
    iteration=$((iteration + 1))
    storage_sync_remote "$source_dir" "$remote_dir"
    local sync_status=$?
    if [ $sync_status -ne 0 ]; then
      echo "[STORAGE] WARN: Remote sync worker pass ${iteration} failed with status ${sync_status}." >&2
      if [ "$abort_on_failure" = true ]; then
        echo "[STORAGE] Remote sync worker detected unreachable destination; notifying renderer to pause." >&2
        if [ -n "$parent_pid" ]; then
          kill -s USR1 "$parent_pid" >/dev/null 2>&1 || true
        fi
        break
      fi
    fi
    if [ -f "$done_flag" ] && [ $sync_status -eq 0 ]; then
      echo "[STORAGE] Remote sync worker confirmed final sync after ${iteration} pass(es)."
      break
    fi
    sleep "$interval_secs"
  done
  echo "[STORAGE] Remote sync worker exiting."
}

start_remote_sync_worker() {
  local source_dir="$1"
  local remote_dir="$2"
  local interval_secs="$3"
  local abort_on_failure="${4:-false}"
  local parent_pid="${5:-}"
  REMOTE_SYNC_DONE_FILE=$(mktemp "${TMPDIR:-/tmp}/remote_sync_done.XXXXXX") || return 1
  rm -f "$REMOTE_SYNC_DONE_FILE"
  remote_sync_worker_loop "$source_dir" "$remote_dir" "$REMOTE_SYNC_DONE_FILE" "$interval_secs" "$abort_on_failure" "$parent_pid" &
  REMOTE_SYNC_WORKER_PID=$!
}

signal_remote_sync_completion() {
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    : > "$REMOTE_SYNC_DONE_FILE"
  fi
}

wait_remote_sync_worker() {
  if [ -n "$REMOTE_SYNC_WORKER_PID" ]; then
    wait "$REMOTE_SYNC_WORKER_PID" || true
    REMOTE_SYNC_WORKER_PID=""
  fi
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    rm -f "$REMOTE_SYNC_DONE_FILE"
    REMOTE_SYNC_DONE_FILE=""
  fi
}

REMOTE_STORAGE_GUARD_PID=""
REMOTE_STORAGE_GUARD_STOP_FILE=""
PARALLEL_GROUP_ID=""

storage_guard_loop() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_bytes="$3"
  local resume_bytes="$4"
  local interval_secs="$5"
  local stop_flag="$6"
  local limit_gb="$7"
  local resume_gb="$8"
  local paused=false
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  while true; do
    if [ -n "$stop_flag" ] && [ -f "$stop_flag" ]; then
      break
    fi
    if [ -z "$pgid" ] || ! kill -0 "$pgid" >/dev/null 2>&1; then
      break
    fi
    local usage_bytes
    usage_bytes=$(du -sb "$watch_dir" 2>/dev/null | awk '{print $1}')
    if [ -z "$usage_bytes" ]; then
      usage_bytes=0
    fi
    if [ "$usage_bytes" -ge "$limit_bytes" ]; then
      if [ "$paused" = false ]; then
        local usage_gb
        usage_gb=$(awk -v b="$usage_bytes" 'BEGIN { printf("%.2f", b / (1024*1024*1024)) }')
        echo "[STORAGE] Local usage ${usage_gb}GB reached limit ${limit_gb}GB; pausing generation until below ${resume_gb}GB."
        kill -STOP -- "-$pgid" >/dev/null 2>&1 || true
        paused=true
      fi
    elif [ "$paused" = true ] && [ "$usage_bytes" -le "$resume_bytes" ]; then
      echo "[STORAGE] Local usage dropped below ${resume_gb}GB; resuming generation."
      kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
      paused=false
    fi
    sleep "$interval_secs"
  done
  if [ "$paused" = true ] && [ -n "$pgid" ]; then
    kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
  fi
}

start_storage_guard() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_gb="$3"
  local resume_gb="$4"
  local interval_secs="$5"
  local limit_bytes=$((limit_gb * 1024 * 1024 * 1024))
  local resume_bytes=$((resume_gb * 1024 * 1024 * 1024))
  REMOTE_STORAGE_GUARD_STOP_FILE=$(mktemp "${TMPDIR:-/tmp}/storage_guard_stop.XXXXXX") || return 1
  rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
  storage_guard_loop "$watch_dir" "$pgid" "$limit_bytes" "$resume_bytes" "$interval_secs" "$REMOTE_STORAGE_GUARD_STOP_FILE" "$limit_gb" "$resume_gb" &
  REMOTE_STORAGE_GUARD_PID=$!
}

stop_storage_guard() {
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    : > "$REMOTE_STORAGE_GUARD_STOP_FILE"
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_PID" ]; then
    wait "$REMOTE_STORAGE_GUARD_PID" || true
    REMOTE_STORAGE_GUARD_PID=""
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
    REMOTE_STORAGE_GUARD_STOP_FILE=""
  fi
}

cleanup_run() {
  wait_remote_sync_worker
  stop_storage_guard
}

# Always tear down helpers even on errors.
trap cleanup_run EXIT
trap handle_remote_storage_unavailable USR1

handle_interrupt() {
  if [ "$STOP_REQUESTED" = true ]; then
    return
  fi
  STOP_REQUESTED=true
  echo "[RUN] Interrupt received; stopping workers..." >&2
  trap - EXIT
  if [ -n "$PARALLEL_GROUP_ID" ]; then
    kill -INT -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
    sleep 1
    kill -TERM -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
  elif [ -n "$PARALLEL_PID" ]; then
    kill -INT "$PARALLEL_PID" >/dev/null 2>&1 || true
    sleep 1
    kill -TERM "$PARALLEL_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$PARALLEL_PID" ]; then
    wait "$PARALLEL_PID" >/dev/null 2>&1 || true
    PARALLEL_PID=""
  fi
  if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    signal_remote_sync_completion
  fi
  cleanup_run
  exit 130
}
trap handle_interrupt INT TERM

# Assignment manifest generation helper (shared implementation lives in scripts/).
generate_assignment_manifest() {
  local exclude_detailed="true"
  if ! storage_bool_true "$EXCLUDE_DETAILED_LABELS"; then
    exclude_detailed="false"
  fi
  CONDA_ENV="${CONDA_ENV}" \
  ACTOR_ROOT="${ACTOR_ROOT}" \
  BAN_LIST="${BAN_LIST}" \
  ASSIGNMENTS_OUT="${ASSIGNMENTS_OUT}" \
  SCENES_DIR="${SCENES_DIR}" \
  TASKS_DIR="${TASKS_DIR}" \
  SEED="${SEED}" \
  EXCLUDE_DETAILED_LABELS="${exclude_detailed}" \
  bash "${SCRIPT_DIR}/scripts/generate_assignment_manifest.sh"
}

if $RESUME_MODE; then
  if [ -n "$RESUME_LOG_PATH" ]; then
    if [ -f "$RESUME_LOG_PATH" ]; then
      RESUME_LOG_PATH="$(abspath "$RESUME_LOG_PATH")"
      echo "[RESUME] Using resume log ${RESUME_LOG_PATH} to skip completed jobs." >&2
    else
      echo "[RESUME] WARN: resume log not found at ${RESUME_LOG_PATH}; continuing with status-json only." >&2
      RESUME_LOG_PATH=""
    fi
  fi
  if [ ! -f "$ASSIGNMENTS_OUT" ]; then
    echo "[RESUME] WARN: Assignment manifest missing at $ASSIGNMENTS_OUT; regenerating."
    generate_assignment_manifest
    if [ ! -f "$ASSIGNMENTS_OUT" ]; then
      echo "[RESUME] ERROR: Failed to regenerate assignment manifest at $ASSIGNMENTS_OUT." >&2
      exit 1
    fi
  fi
  echo "[RESUME] Using manifest $ASSIGNMENTS_OUT"
  CLEAR_LOCAL_OUTPUT_DIR=false
fi

# Utility: guarantee output dir exists + is writable before we drop a ton of
# frames in there.
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

# Optionally wipe previous contents to keep runs deterministic unless resume
# mode disabled the cleanup step earlier.
prepare_local_output_dir() {
  local target="$1"
  ensure_writable_dir "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

prepare_local_output_dir "$OUTPUT_DIR"

# Connectivity sanity check so we fail early if the NAS is unreachable before any
# heavy compute starts.
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

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  if ! storage_test_remote_connection "$REMOTE_TARGET_DIR"; then
    if [ "$REMOTE_ONLY_STORAGE" = true ]; then
      echo "[CHECK] ERROR: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable and no alternate storage is configured." >&2
      exit 1
    else
      echo "[CHECK] WARN: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable; continuing with other storage backends." >&2
    fi
  else
    echo "[CHECK] Remote destination reachable at ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR}"
  fi
fi

echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"


# Assignment planning is deterministic: in resume mode we skip generation and
# reuse the previous manifest so scene/actor pairings stay stable.
if $RESUME_MODE; then
  echo "[RESUME] Skipping assignment generation and reusing ${ASSIGNMENTS_OUT}"
else
  generate_assignment_manifest
fi

# Rendering CLI snippets are composed here so storage flags can extend/override
# behavior (NAS uploads, BEV toggles, etc.) without duplicating the Python call.
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  render_extra_snippets+=("--offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb ${OFFLOAD_MIN_FREE_GB}")
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" python parallel_render_paths_telesim.py
  --assignment-manifest "${ASSIGNMENTS_OUT}"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --output-dir "${OUTPUT_DIR}"
  --progress-json "${PROGRESS_JSON}"
  --status-json "${STATUS_JSON}"
  --per-job-metrics-dir "${PER_JOB_METRICS_DIR}"
  --report-out "${PARALLEL_REPORT_DIR}"
  --cuda-oom-retry-delay "${CUDA_OOM_RETRY_DELAY}"
  --cuda-oom-max-retries "${CUDA_OOM_MAX_RETRIES}"
)
if storage_bool_true "$EXCLUDE_DETAILED_LABELS"; then
  parallel_cmd+=(--exclude-detailed-labels)
else
  parallel_cmd+=(--no-exclude-detailed-labels)
fi
if $RESUME_MODE && [ -n "$RESUME_LOG_PATH" ]; then
  parallel_cmd+=(--skip-completed-log "$RESUME_LOG_PATH")
fi
if storage_bool_true "$RETRY_CUDA_OOM"; then
  parallel_cmd+=(--retry-cuda-oom)
else
  parallel_cmd+=(--no-retry-cuda-oom)
fi
# Thread the resume log into the renderer so it can skip completed scene/actor
# pairs. Remaining CLI snippets (overwrite/offload/etc.) are appended below.
for snippet in "${render_extra_snippets[@]}"; do
  parallel_cmd+=(--render-extra-args "$snippet")
done

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  remote_abort_flag="false"
  if [ "$REMOTE_ONLY_STORAGE" = true ]; then
    remote_abort_flag="true"
  fi
  start_remote_sync_worker "$OUTPUT_DIR" "$REMOTE_TARGET_DIR" "$REMOTE_SYNC_INTERVAL_SECS" "$remote_abort_flag" "$$"
fi

render_status=0
set +e
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ]; then
  "$SETSID_BIN" "${parallel_cmd[@]}" &
else
  "${parallel_cmd[@]}" &
fi
PARALLEL_PID=$!
PARALLEL_GROUP_ID="$PARALLEL_PID"
if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ]; then
  if ! start_storage_guard "$OUTPUT_DIR" "$PARALLEL_GROUP_ID" "$REMOTE_STORAGE_LIMIT_GB" "$REMOTE_STORAGE_RESUME_GB" "$REMOTE_STORAGE_GUARD_INTERVAL_SECS"; then
    echo "[STORAGE] WARN: Failed to start storage guard; continuing without throttling." >&2
    REMOTE_STORAGE_GUARD_ENABLED=false
    stop_storage_guard
  fi
fi
wait "$PARALLEL_PID"
render_status=$?
PARALLEL_PID=""
PARALLEL_GROUP_ID=""
set -e
if [ "$REMOTE_STORAGE_UNAVAILABLE" = true ]; then
  render_status=99
  echo "[STORAGE] Remote destination unavailable; generation paused. Resume once storage is back online." >&2
elif [ $render_status -ne 0 ]; then
  echo "[WARN] parallel_render_paths.py exited with status ${render_status}, continuing per request."
fi

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  if [ "$REMOTE_STORAGE_UNAVAILABLE" = false ]; then
    signal_remote_sync_completion
  fi
  wait_remote_sync_worker
fi

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" && [ "$REMOTE_STORAGE_UNAVAILABLE" = false ]; then
  # When purely offloading to NAS/remote, purge local outputs to conserve disk.
  if [ -d "$OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
  fi
fi

exit $render_status
