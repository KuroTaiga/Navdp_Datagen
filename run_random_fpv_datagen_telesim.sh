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
SCENES_DIR=${SCENES_DIR:-./data/CHINGMU_scenes_rescaled}
TASKS_DIR=${TASKS_DIR:-./data/CHINGMU_75_rescaled_0800_42_iter1}
OUTPUT_DIR=${OUTPUT_DIR:-./navdata/CHINGMU_0800}
PROGRESS_JSON=${PROGRESS_JSON:-./analysis/CHINGMU_0800_progress.json}
STATUS_JSON=${STATUS_JSON:-./analysis/CHINGMU_0800_status.json}
PER_JOB_METRICS_DIR=${PER_JOB_METRICS_DIR:-./analysis/fpv_telesim_metrics}
PARALLEL_REPORT_DIR=${PARALLEL_REPORT_DIR:-./parallel_render_report_CHINGMU_0800_telesim.json}
ERROR_LOG=${ERROR_LOG:-./CHINGMU_0800_telesim.log}
WORKERS=${WORKERS:-24}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-0}
FPV_FOLLOW_DISTANCE=${FPV_FOLLOW_DISTANCE:-0}
RESUME_MODE=${RESUME_MODE:-false}
RESUME_LOG_PATH=${RESUME_LOG_PATH:-}
RETRY_CUDA_OOM=${RETRY_CUDA_OOM:-true}
CUDA_OOM_RETRY_DELAY=${CUDA_OOM_RETRY_DELAY:-10}
CUDA_OOM_MAX_RETRIES=${CUDA_OOM_MAX_RETRIES:--1}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}

ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-false}
ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-false}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-false}
ENABLE_CAMERA_METADATA=${ENABLE_CAMERA_METADATA:-true}
ENABLE_FOLLOW_METADATA=${ENABLE_FOLLOW_METADATA:-false}
EXCLUDE_DETAILED_LABELS=${EXCLUDE_DETAILED_LABELS:-true}
WORKER_PROGRESS=${WORKER_PROGRESS:-false}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}
ANTIALIASING=${ANTIALIASING:-false}
MAX_LABELS=${MAX_LABELS:-}
SH_DEGREE=${SH_DEGREE:--1}
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
CL_NORMAL_FILTER=${CL_NORMAL_FILTER:-box}
CL_NORMAL_KERNEL=${CL_NORMAL_KERNEL:-2}
CL_NORMAL_SIGMA_RANGE=${CL_NORMAL_SIGMA_RANGE:-0.1}
CL_NORMAL_SIGMA_DOMAIN=${CL_NORMAL_SIGMA_DOMAIN:-1.0}
CL_SHADOW=${CL_SHADOW:-false}
CL_SHADOW_BIAS=${CL_SHADOW_BIAS:-0.02}
CL_SHADOW_STRENGTH=${CL_SHADOW_STRENGTH:-0.2}
CL_SHADOW_PCF=${CL_SHADOW_PCF:-0}
CL_SHADOW_COMPARE=${CL_SHADOW_COMPARE:-z}

# Unsupported features in TeleSim pipeline (warn if enabled)
NPC_ENABLE=${NPC_ENABLE:-false}
LIGHT_MODE=${LIGHT_MODE:-none}
CL_ENABLE=${CL_ENABLE:-false}
if [ "${NPC_ENABLE}" != "false" ] || [ "${LIGHT_MODE}" != "none" ] || [ "${CL_ENABLE}" != "false" ]; then
  echo "[WARN] TeleSim pipeline ignores NPC/light/CL settings." >&2
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
if [ -n "${LIGHT_MODE:-}" ] && [ "${LIGHT_MODE}" != "none" ]; then
  render_extra_args+=" --light-mode ${LIGHT_MODE}"
  render_extra_args+=" --light-strength ${LIGHT_STRENGTH:-0.0}"
  render_extra_args+=" --light-radius ${LIGHT_RADIUS:-0.45}"
  render_extra_args+=" --light-center ${LIGHT_CENTER_X:-0.5} ${LIGHT_CENTER_Y:-0.5}"
  render_extra_args+=" --light-jitter ${LIGHT_JITTER:-0.0}"
  render_extra_args+=" --light-temp-k ${LIGHT_TEMP_K:-0}"
  render_extra_args+=" --light-vignette ${LIGHT_VIGNETTE:-0.0}"
  render_extra_args+=" --light-seed ${LIGHT_SEED:-0}"
fi
if [ "${CL_ENABLE}" = "true" ]; then
  render_extra_args+=" --cl-enable"
  render_extra_args+=" --cl-strength ${CL_STRENGTH:-1.0}"
  render_extra_args+=" --cl-color ${CL_COLOR_R:-1.0} ${CL_COLOR_G:-1.0} ${CL_COLOR_B:-1.0}"
  render_extra_args+=" --cl-ambient ${CL_AMBIENT:-0.2}"
  render_extra_args+=" --cl-diffuse ${CL_DIFFUSE:-1.0}"
  render_extra_args+=" --cl-specular ${CL_SPECULAR:-0.2}"
  render_extra_args+=" --cl-shininess ${CL_SHININESS:-16.0}"
  render_extra_args+=" --cl-range ${CL_RANGE:-0.0}"
  render_extra_args+=" --cl-offset ${CL_OFFSET_X:-0.0} ${CL_OFFSET_Y:-0.0} ${CL_OFFSET_Z:-0.0}"
  render_extra_args+=" --cl-normal-smooth ${CL_NORMAL_SMOOTH:-0}"
  render_extra_args+=" --cl-normal-filter ${CL_NORMAL_FILTER:-box}"
  render_extra_args+=" --cl-normal-kernel ${CL_NORMAL_KERNEL:-2}"
  render_extra_args+=" --cl-normal-sigma-range ${CL_NORMAL_SIGMA_RANGE:-0.1}"
  render_extra_args+=" --cl-normal-sigma-domain ${CL_NORMAL_SIGMA_DOMAIN:-1.0}"
  if [ "${CL_SHADOW:-false}" = "true" ]; then
    render_extra_args+=" --cl-shadow"
  else
    render_extra_args+=" --no-cl-shadow"
  fi
  render_extra_args+=" --cl-shadow-bias ${CL_SHADOW_BIAS:-0.02}"
  render_extra_args+=" --cl-shadow-strength ${CL_SHADOW_STRENGTH:-0.2}"
  render_extra_args+=" --cl-shadow-pcf ${CL_SHADOW_PCF:-0}"
  render_extra_args+=" --cl-shadow-compare ${CL_SHADOW_COMPARE:-z}"
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN" "$SCRIPT_DIR/parallel_render_paths_telesim.py"
  --fpv-only
  --fpv-follow-distance "${FPV_FOLLOW_DISTANCE}"
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
if [ "${WORKER_PROGRESS}" = "true" ]; then
  parallel_cmd+=(--worker-progress)
fi
parallel_cmd+=(--render-extra-args "$render_extra_args")

"${parallel_cmd[@]}"
