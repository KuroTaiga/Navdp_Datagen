#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

# Keep Python stdout unbuffered so progress lines show up in logs.
export PYTHONUNBUFFERED=1

show_usage_and_exit() {
  echo "Usage: $(basename "$0") [RESUME [LOG_PATH]] [--pipeline gpu|legacy] [--legacy-pipeline]" >&2
  exit 1
}

PIPELINE_MODE=${PIPELINE_MODE:-legacy}
RESUME_MODE=true
RESUME_LOG_PATH="CHINGMU_0800_new.log"
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

abspath() {
  "$PYTHON_BIN" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# -----------------------------------------------------------------------------
# User-configurable defaults (override via env vars before invoking the script).
# -----------------------------------------------------------------------------
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

CONDA_ENV=${CONDA_ENV:-cuda121}
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_scenes_rescaled}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./navdata/CHINGMU_0800}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/npc_dataset_10w}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
PROGRESS_JSON=${PROGRESS_JSON:-./analysis/CHINGMU_0800_progress.json}
STATUS_JSON=${STATUS_JSON:-./analysis/CHINGMU_0800_status.json}
PER_JOB_METRICS_DIR=${PER_JOB_METRICS_DIR:-./analysis/fpv_npc_metrics}
PARALLEL_REPORT_DIR=${PARALLEL_REPORT_DIR:-./parallel_render_report_CHINGMU_0800.json}
ERROR_LOG=${ERROR_LOG:-./CHINGMU_0800.log}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/baicj-telenav}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-lixinhai@root@ssh-34.default@58.59.115.26}
: "${REMOTE_SSH_PORT:=30022}"
LOCAL_OUTPUT_BASENAME="$(basename "$OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"
REMOTE_SYNC_INTERVAL_SECS=${REMOTE_SYNC_INTERVAL_SECS:-120}
WORKERS=${WORKERS:-24}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-0}
FPV_FOLLOW_DISTANCE=${FPV_FOLLOW_DISTANCE:-0}

# Prefer the ffmpeg inside CONDA_ENV for NVENC, since the caller shell may be in a different env.
if [ "$PIPELINE_MODE" = "gpu" ] && [ -z "${FFMPEG_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
  PATH_FFMPEG=$(command -v ffmpeg 2>/dev/null || true)
  if [ -z "${IMAGEIO_FFMPEG_EXE:-}" ] || [ "${IMAGEIO_FFMPEG_EXE}" = "${PATH_FFMPEG}" ]; then
    FFMPEG_IN_ENV=$(conda run --no-capture-output -n "$CONDA_ENV" which ffmpeg 2>/dev/null || true)
    if [ -n "${FFMPEG_IN_ENV}" ]; then
      export IMAGEIO_FFMPEG_EXE="${FFMPEG_IN_ENV}"
    fi
  fi
fi

# Robot camera stats
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3} #1.3m

# Optional NPC placement/debug. Leave values empty to skip.
NPC_ENABLE=${NPC_ENABLE:-false}
NPC_DENSITY_COVERAGE=${NPC_DENSITY_COVERAGE:-0.8}
NPC_COUNT=${NPC_COUNT:-20}
NPC_MAX_COUNT=${NPC_MAX_COUNT:-20}
NPC_PRIORITY=${NPC_PRIORITY:-coverage}
NPC_MAX_RANGE=${NPC_MAX_RANGE:-15}
NPC_FREE_THRESHOLD=${NPC_FREE_THRESHOLD:-250}
NPC_FREE_WHITE=${NPC_FREE_WHITE:-true}
NPC_DENSITY_MODE=${NPC_DENSITY_MODE:-angular}
NPC_ZONE_RATIO=${NPC_ZONE_RATIO:-1:2:1}
NPC_ROTATE_MASK_180=${NPC_ROTATE_MASK_180:-true} # rotate to aligne with actual locaiton of world coordinates. 
NPC_MIN_DISTANCE=${NPC_MIN_DISTANCE:-2.5}
NPC_EXTRA_FLAGS=${NPC_EXTRA_FLAGS:-}
NPC_AUTO_CLEARANCE=${NPC_AUTO_CLEARANCE:-true}
NPC_FRAME_POOL_SIZE=${NPC_FRAME_POOL_SIZE:-30} # pool size 30 is a good middle ground
NPC_PLACEMENT_BACKEND=${NPC_PLACEMENT_BACKEND:-}
ACTOR_ROOT=${ACTOR_ROOT:-./data/SHHQ_gs/walking}

RETRY_CUDA_OOM=${RETRY_CUDA_OOM:-true}
CUDA_OOM_RETRY_DELAY=${CUDA_OOM_RETRY_DELAY:-10}
CUDA_OOM_MAX_RETRIES=${CUDA_OOM_MAX_RETRIES:--1}

# To enable per-path BEV debug images, run: ENABLE_BEV_IMAGES=true ./run_random_fpv_datagen.sh
ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-false}
ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-false}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-false}
ENABLE_CAMERA_METADATA=${ENABLE_CAMERA_METADATA:-true}
ENABLE_FOLLOW_METADATA=${ENABLE_FOLLOW_METADATA:-false}
VERBOSE=${VERBOSE:-true}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
WORKER_PROGRESS=${WORKER_PROGRESS:-false}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}
ANTIALIASING=${ANTIALIASING:-false}
MAX_LABELS=${MAX_LABELS:-}
SH_DEGREE=${SH_DEGREE:--1}

GPU_ONLY_FLAG="--gpu-only"
if [ "$PIPELINE_MODE" = "legacy" ]; then
  : "${PLY_TRANSFORM_BACKEND:=gpu}" # trying GPU to test if there is some performance imporvements
  : "${VIDEO_BACKEND:=cpu}" # cpu=libx264, nvenc=ffmpeg h264_nvenc, gpu=PyNvVideoCodec (pynvcodec)
  : "${NPC_PLACEMENT_BACKEND:=cpu}" #was set to GPU but the imapct is small so CPU is fine
else
  : "${PLY_TRANSFORM_BACKEND:=gpu}"
  : "${VIDEO_BACKEND:=nvenc}" # cpu=libx264, nvenc=ffmpeg h264_nvenc, gpu=PyNvVideoCodec (pynvcodec)
  : "${NPC_PLACEMENT_BACKEND:=gpu}"
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

#for changing lighting levels. We can also use the script under ./lighting folder to directly modify based on the mp4 results
LIGHT_MODE=${LIGHT_MODE:-none}
LIGHT_STRENGTH=${LIGHT_STRENGTH:-0.0}
LIGHT_RADIUS=${LIGHT_RADIUS:-0.45}
LIGHT_CENTER_X=${LIGHT_CENTER_X:-0.5}
LIGHT_CENTER_Y=${LIGHT_CENTER_Y:-0.5}
LIGHT_JITTER=${LIGHT_JITTER:-0.0}
LIGHT_TEMP_K=${LIGHT_TEMP_K:-0}
LIGHT_VIGNETTE=${LIGHT_VIGNETTE:-0.0}
LIGHT_SEED=${LIGHT_SEED:-0}
CL_ENABLE=${CL_ENABLE:-false}
CL_STRENGTH=${CL_STRENGTH:-1.0}
CL_COLOR_R=${CL_COLOR_R:-1.0}
CL_COLOR_G=${CL_COLOR_G:-1.0}
CL_COLOR_B=${CL_COLOR_B:-1.0}
CL_AMBIENT=${CL_AMBIENT:-0.2}
CL_DIFFUSE=${CL_DIFFUSE:-1.0}
CL_SPECULAR=${CL_SPECULAR:-0.2}
CL_SHININESS=${CL_SHININESS:-16.0}
CL_RANGE=${CL_RANGE:-0.0}
CL_OFFSET_X=${CL_OFFSET_X:-0.0}
CL_OFFSET_Y=${CL_OFFSET_Y:-0.0}
CL_OFFSET_Z=${CL_OFFSET_Z:-0.0}
CL_NORMAL_SMOOTH=${CL_NORMAL_SMOOTH:-0}
CL_SHADOW=${CL_SHADOW:-false}
CL_SHADOW_BIAS=${CL_SHADOW_BIAS:-0.02}
CL_SHADOW_STRENGTH=${CL_SHADOW_STRENGTH:-0.2}
CL_SHADOW_PCF=${CL_SHADOW_PCF:-0}

render_extra_args="--overwrite --stabilize ${GPU_ONLY_FLAG} --view-mode forward --height-offset ${HEIGHT_OFFSET}"
# render_extra_args+=" --navdp-ply-per-scene  --no-validate-path-bounds"
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
if storage_bool_true "$VERBOSE"; then
  render_extra_args+=' --verbose'
fi
if storage_bool_true "$ANTIALIASING"; then
  render_extra_args+=' --antialiasing'
else
  render_extra_args+=' --no-antialiasing'
fi
if [ -n "${MAX_LABELS}" ]; then
  render_extra_args+=" --max-labels ${MAX_LABELS}"
fi
if [ -n "${SH_DEGREE}" ]; then
  render_extra_args+=" --sh-degree ${SH_DEGREE}"
fi
if [ "${LIGHT_MODE}" != "none" ]; then
  render_extra_args+=" --light-mode ${LIGHT_MODE}"
  render_extra_args+=" --light-strength ${LIGHT_STRENGTH}"
  render_extra_args+=" --light-radius ${LIGHT_RADIUS}"
  render_extra_args+=" --light-center ${LIGHT_CENTER_X} ${LIGHT_CENTER_Y}"
  render_extra_args+=" --light-jitter ${LIGHT_JITTER}"
  render_extra_args+=" --light-temp-k ${LIGHT_TEMP_K}"
  render_extra_args+=" --light-vignette ${LIGHT_VIGNETTE}"
  render_extra_args+=" --light-seed ${LIGHT_SEED}"
fi
if storage_bool_true "$CL_ENABLE"; then
  render_extra_args+=" --cl-enable"
  render_extra_args+=" --cl-strength ${CL_STRENGTH}"
  render_extra_args+=" --cl-color ${CL_COLOR_R} ${CL_COLOR_G} ${CL_COLOR_B}"
  render_extra_args+=" --cl-ambient ${CL_AMBIENT}"
  render_extra_args+=" --cl-diffuse ${CL_DIFFUSE}"
  render_extra_args+=" --cl-specular ${CL_SPECULAR}"
  render_extra_args+=" --cl-shininess ${CL_SHININESS}"
  render_extra_args+=" --cl-range ${CL_RANGE}"
  render_extra_args+=" --cl-offset ${CL_OFFSET_X} ${CL_OFFSET_Y} ${CL_OFFSET_Z}"
  render_extra_args+=" --cl-normal-smooth ${CL_NORMAL_SMOOTH}"
  render_extra_args+=" --cl-shadow-bias ${CL_SHADOW_BIAS}"
  render_extra_args+=" --cl-shadow-strength ${CL_SHADOW_STRENGTH}"
  render_extra_args+=" --cl-shadow-pcf ${CL_SHADOW_PCF}"
  if storage_bool_true "$CL_SHADOW"; then
    render_extra_args+=" --cl-shadow"
  fi
fi
render_extra_snippets=("$render_extra_args")

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
  if storage_bool_true "$NPC_ROTATE_MASK_180"; then
    npc_args+=("--npc-rotate-mask-180")
  fi
  if [ -n "${NPC_DENSITY_COVERAGE:-}" ]; then
    npc_args+=("--npc-density-coverage ${NPC_DENSITY_COVERAGE}")
  fi
  if [ -n "${NPC_COUNT:-}" ]; then
    npc_args+=("--npc-count ${NPC_COUNT}")
  fi
  if [ -n "${NPC_MAX_COUNT:-}" ]; then
    npc_args+=("--npc-max-count ${NPC_MAX_COUNT}")
  fi
  if [ -n "${NPC_MAX_RANGE:-}" ]; then
    npc_args+=("--npc-max-range ${NPC_MAX_RANGE}")
  fi
  # if [ -n "${NPC_MIN_DISTANCE:-}" ]; then
  #   npc_args+=("--npc-min-distance ${NPC_MIN_DISTANCE}") # telesim
  # fi
  if [ -n "${NPC_FRAME_POOL_SIZE:-}" ]; then
    npc_args+=("--npc-frame-pool-size ${NPC_FRAME_POOL_SIZE}")
  fi
  # if [ -n "${NPC_PLACEMENT_BACKEND:-}" ]; then # for telesim
    # npc_args+=("--npc-placement-backend ${NPC_PLACEMENT_BACKEND}")
  # fi
  if storage_bool_true "$NPC_AUTO_CLEARANCE"; then
    if [ -d "$ACTOR_ROOT" ]; then
      npc_args+=("--npc-auto-clearance" "--npc-actor-root ${ACTOR_ROOT}")
    else
      echo "[NPC] WARN: ACTOR_ROOT ${ACTOR_ROOT} not found; skipping auto-clearance." >&2
    fi
  fi
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
  CLEAR_LOCAL_OUTPUT_DIR=false
fi

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


if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  render_extra_snippets+=("--offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb ${OFFLOAD_MIN_FREE_GB}")
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" python parallel_render_paths_telesim.py
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --output-dir "${OUTPUT_DIR}"
  --report-out "${PARALLEL_REPORT_DIR}"
)
#  --fpv-only   --fpv-follow-distance "${FPV_FOLLOW_DISTANCE}"   --minimal-frames "${MINIMAL_FRAMES}"
# --progress-json "${PROGRESS_JSON}"
# --status-json "${STATUS_JSON}"
# --error-log "${ERROR_LOG}"   --per-job-metrics-dir "${PER_JOB_METRICS_DIR}"
# --cuda-oom-retry-delay "${CUDA_OOM_RETRY_DELAY}"
# --cuda-oom-max-retries "${CUDA_OOM_MAX_RETRIES}"
if storage_bool_true "$WORKER_PROGRESS"; then
  parallel_cmd+=(--worker-progress)
fi
# if storage_bool_true "$EXCLUDE_DETAILED_LABELS"; then
#   parallel_cmd+=(--exclude-detailed-labels)
# else
#   parallel_cmd+=(--no-exclude-detailed-labels)
# fi
# if $RESUME_MODE && [ -n "$RESUME_LOG_PATH" ]; then
#   parallel_cmd+=(--skip-completed-log "$RESUME_LOG_PATH")
# fi
# if storage_bool_true "$RETRY_CUDA_OOM"; then
#   parallel_cmd+=(--retry-cuda-oom)
# else
#   parallel_cmd+=(--no-retry-cuda-oom)
# fi
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
  if [ -d "$OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
  fi
fi

exit $render_status
