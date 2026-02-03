#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

# User-configurable defaults (override via env vars)
CONDA_ENV=${CONDA_ENV:-cuda121}
USE_CONDA_RUN=${USE_CONDA_RUN:-auto}
SCENE_ID=${SCENE_ID:-}
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_scenes_rescaled}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./navdata/CHINGMU_0800_follow_flaw}
PROGRESS_JSON=${PROGRESS_JSON:-./analysis/CHINGMU_0800_follow_1.json}
STATUS_JSON=${STATUS_JSON:-./analysis/CHINGMU_0800_follow_status.json}
PER_JOB_METRICS_DIR=${PER_JOB_METRICS_DIR:-./analysis/CHINGMU_0800_follow_metrics_telesim}
PARALLEL_REPORT_DIR=${PARALLEL_REPORT_DIR:-./parallel_render_report_CHINGMU_follow_telesim.json}
ERROR_LOG=${ERROR_LOG:-./CHINGMU_0800_follow_telesim.log}
WORKERS=${WORKERS:-24}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-38}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_CHINGMU.json}
RESUME_LOG_PATH=${RESUME_LOG_PATH:-}
RESUME_MODE=${RESUME_MODE:-false}
RETRY_CUDA_OOM=${RETRY_CUDA_OOM:-true}
CUDA_OOM_RETRY_DELAY=${CUDA_OOM_RETRY_DELAY:-10}
CUDA_OOM_MAX_RETRIES=${CUDA_OOM_MAX_RETRIES:--1}

ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-false}
ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-false}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-false}
ENABLE_CAMERA_METADATA=${ENABLE_CAMERA_METADATA:-true}
ENABLE_FOLLOW_METADATA=${ENABLE_FOLLOW_METADATA:-true}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}
ANTIALIASING=${ANTIALIASING:-false}
MAX_LABELS=${MAX_LABELS:-}
SH_DEGREE=${SH_DEGREE:--1}

# Unsupported features in TeleSim pipeline (warn if enabled)
ACTOR_ROOT=${ACTOR_ROOT:-}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-}
NPC_ENABLE=${NPC_ENABLE:-false}
LIGHT_MODE=${LIGHT_MODE:-none}
CL_ENABLE=${CL_ENABLE:-false}
if [ "${NPC_ENABLE}" != "false" ] || [ "${LIGHT_MODE}" != "none" ] || [ "${CL_ENABLE}" != "false" ]; then
  echo "[WARN] TeleSim pipeline ignores NPC/lighting/CL settings (actor follow is supported)." >&2
fi

render_extra_args="--overwrite --stabilize --height-offset ${HEIGHT_OFFSET}"
if [ -n "${MAX_LABELS}" ]; then
  render_extra_args+=" --max-labels ${MAX_LABELS}"
fi
if [ "${ENABLE_BEV_IMAGES}" = "true" ]; then
  render_extra_args+=' --show-BEV'
else
  render_extra_args+=' --no-show-BEV'
fi
if [ "${ENABLE_VIDEO_OUTPUT}" = "true" ]; then
  render_extra_args+=' --video'
else
  render_extra_args+=' --no-video'
fi
if [ -n "${VIDEO_NVENC_PRESET}" ]; then
  render_extra_args+=" --video-nvenc-preset ${VIDEO_NVENC_PRESET}"
fi
if [ -n "${VIDEO_NVENC_BITRATE}" ]; then
  render_extra_args+=" --video-nvenc-bitrate ${VIDEO_NVENC_BITRATE}"
fi
if [ "${ENABLE_RGB_FRAMES}" = "true" ]; then
  render_extra_args+=' --rgb-frames'
else
  render_extra_args+=' --no-rgb-frames'
fi
if [ "${ENABLE_DEPTH_OUTPUT}" = "true" ]; then
  render_extra_args+=' --save-depth-maps'
else
  render_extra_args+=' --no-save-depth-maps'
fi
if [ "${ENABLE_CAMERA_METADATA}" = "true" ]; then
  render_extra_args+=' --save-camera-metadata'
else
  render_extra_args+=' --no-save-camera-metadata'
fi
if [ "${ENABLE_FOLLOW_METADATA}" = "true" ]; then
  render_extra_args+=' --save-follow-metadata'
else
  render_extra_args+=' --no-save-follow-metadata'
fi
if [ "${ANTIALIASING}" = "true" ]; then
  render_extra_args+=' --antialiasing'
else
  render_extra_args+=' --no-antialiasing'
fi
if [ "${SH_DEGREE}" != "-1" ]; then
  render_extra_args+=" --sh-degree ${SH_DEGREE}"
fi

parallel_cmd=()
if [ "${USE_CONDA_RUN}" = "auto" ]; then
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV}" ]; then
    USE_CONDA_RUN="false"
  else
    USE_CONDA_RUN="true"
  fi
fi

if [ "${USE_CONDA_RUN}" = "true" ]; then
  parallel_cmd+=(conda run --no-capture-output -n "$CONDA_ENV")
fi
parallel_cmd+=(
  "$PYTHON_BIN" "$SCRIPT_DIR/parallel_render_paths_telesim.py"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --output-dir "${OUTPUT_DIR}"
  --error-log "${ERROR_LOG}"
  --progress-json "${PROGRESS_JSON}"
  --status-json "${STATUS_JSON}"
  --per-job-metrics-dir "${PER_JOB_METRICS_DIR}"
  --report-out "${PARALLEL_REPORT_DIR}"
)
if [ -n "${SCENE_ID}" ]; then
  parallel_cmd+=(--scene "${SCENE_ID}")
fi
if [ -n "${ASSIGNMENTS_OUT}" ]; then
  parallel_cmd+=(--assignment-manifest "${ASSIGNMENTS_OUT}")
fi
if [ "${RESUME_MODE}" = "true" ]; then
  parallel_cmd+=(--resume)
fi
if [ -n "${RESUME_LOG_PATH}" ]; then
  parallel_cmd+=(--skip-completed-log "${RESUME_LOG_PATH}")
fi
if [ "${RETRY_CUDA_OOM}" = "true" ]; then
  parallel_cmd+=(--retry-cuda-oom)
else
  parallel_cmd+=(--no-retry-cuda-oom)
fi
parallel_cmd+=(--cuda-oom-retry-delay "${CUDA_OOM_RETRY_DELAY}")
parallel_cmd+=(--cuda-oom-max-retries "${CUDA_OOM_MAX_RETRIES}")
if [ "${EXCLUDE_DETAILED_LABELS}" = "true" ]; then
  parallel_cmd+=(--exclude-detailed-labels)
else
  parallel_cmd+=(--no-exclude-detailed-labels)
fi
parallel_cmd+=(--render-extra-args "$render_extra_args")

PARALLEL_PID=""
on_interrupt() {
  if [ -n "${PARALLEL_PID}" ]; then
    kill -INT -- "-${PARALLEL_PID}" 2>/dev/null || kill -INT "${PARALLEL_PID}" 2>/dev/null || true
    wait "${PARALLEL_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap on_interrupt INT TERM

if command -v setsid >/dev/null 2>&1; then
  setsid "${parallel_cmd[@]}" &
else
  "${parallel_cmd[@]}" &
fi
PARALLEL_PID=$!
set +e
wait "${PARALLEL_PID}"
RC=$?
set -e
trap - INT TERM
exit $RC
