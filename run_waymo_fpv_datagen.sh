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

CONDA_ENV=${CONDA_ENV:-cuda121}
USE_CONDA=${USE_CONDA:-true}
RUNNER=("$PYTHON_BIN")
if [ "$USE_CONDA" = "true" ] && command -v conda >/dev/null 2>&1; then
  RUNNER=(conda run -n "$CONDA_ENV" "$PYTHON_BIN")
fi

FFMPEG_BIN=${FFMPEG_BIN:-}
if [ -n "$FFMPEG_BIN" ]; then
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
elif command -v ffmpeg >/dev/null 2>&1; then
  export IMAGEIO_FFMPEG_EXE
  IMAGEIO_FFMPEG_EXE=$(command -v ffmpeg)
fi

SCENES_DIR=${SCENES_DIR:-./waymo_scenes}
TASKS_DIR=${TASKS_DIR:-./waymo_tasks}
OUTPUT_DIR=${OUTPUT_DIR:-./data1/waymo_demo}
SCENE_ID=${SCENE_ID:-0001_000001}

VIEW_MODE=${VIEW_MODE:-forward}
HEIGHT_OFFSET=${HEIGHT_OFFSET:--1}
PATH_HANDEDNESS=${PATH_HANDEDNESS:-right}
PLY_TRANSFORM_BACKEND=${PLY_TRANSFORM_BACKEND:-gpu}
VIDEO_BACKEND=${VIDEO_BACKEND:-gpu}

ENABLE_VIDEO_OUTPUT=${ENABLE_VIDEO_OUTPUT:-true}
ENABLE_RGB_FRAMES=${ENABLE_RGB_FRAMES:-true}
ENABLE_DEPTH_OUTPUT=${ENABLE_DEPTH_OUTPUT:-true}
ENABLE_CAMERA_METADATA=${ENABLE_CAMERA_METADATA:-true}
ENABLE_BEV_IMAGES=${ENABLE_BEV_IMAGES:-true}
ENABLE_MIRROR_TRANSLATION=${ENABLE_MIRROR_TRANSLATION:-true}
GPU_ONLY=${GPU_ONLY:-true}
DEBUG=${DEBUG:-true}
SKIP_SUMMARY=${SKIP_SUMMARY:-true}
CAMERA_HUMAN_PLY_DIR=${CAMERA_HUMAN_PLY_DIR:-./data/human_gs_source}
CAMERA_HUMAN_HEIGHT=${CAMERA_HUMAN_HEIGHT:-1.7}
CAMERA_HUMAN_SEED=${CAMERA_HUMAN_SEED:-0}
CAMERA_HUMAN_STRIDE=${CAMERA_HUMAN_STRIDE:-1}
CAMERA_HUMAN_MAX_FRAMES=${CAMERA_HUMAN_MAX_FRAMES:-}
REVERSE_FORWARD=${REVERSE_FORWARD:-false}
NEGATE_RASTER_WORLD_XY=${NEGATE_RASTER_WORLD_XY:-true}

render_args=(
  --scenes-dir "$SCENES_DIR"
  --tasks-dir "$TASKS_DIR"
  --output-dir "$OUTPUT_DIR"
  --view-mode "$VIEW_MODE"
  --height-offset "$HEIGHT_OFFSET"
  --path-handedness "$PATH_HANDEDNESS"
  --ply-transform-backend "$PLY_TRANSFORM_BACKEND"
  --video-backend "$VIDEO_BACKEND"
)

if [ -z "$SCENE_ID" ]; then
  if [ -d "$TASKS_DIR" ]; then
    SCENE_ID=$(find "$TASKS_DIR" -maxdepth 1 -mindepth 1 -type d | sort | head -n 1 | xargs -n1 basename)
  fi
fi
if [ -n "$SCENE_ID" ]; then
  render_args+=(--scene "$SCENE_ID")
else
  echo "[WAYMO] ERROR: No scene found under $TASKS_DIR; set SCENE_ID manually." >&2
  exit 1
fi

if [ "$GPU_ONLY" = "true" ]; then
  render_args+=(--gpu-only)
fi
if [ "$DEBUG" = "true" ]; then
  render_args+=(--debug)
fi
if [ "$ENABLE_BEV_IMAGES" = "true" ]; then
  render_args+=(--show-BEV)
else
  render_args+=(--no-show-BEV)
fi
if [ "$ENABLE_VIDEO_OUTPUT" = "true" ]; then
  render_args+=(--video)
else
  render_args+=(--no-video)
fi
if [ "$ENABLE_RGB_FRAMES" = "true" ]; then
  render_args+=(--rgb-frames)
else
  render_args+=(--no-rgb-frames)
fi
if [ "$ENABLE_DEPTH_OUTPUT" = "true" ]; then
  render_args+=(--save-depth-maps)
else
  render_args+=(--no-save-depth-maps)
fi
if [ "$ENABLE_CAMERA_METADATA" = "true" ]; then
  render_args+=(--save-camera-metadata)
else
  render_args+=(--no-save-camera-metadata)
fi
if [ "$ENABLE_MIRROR_TRANSLATION" != "true" ]; then
  render_args+=(--no-mirror-translation)
fi
if [ "$SKIP_SUMMARY" = "true" ]; then
  render_args+=(--skip-summary)
fi
if [ -n "$CAMERA_HUMAN_PLY_DIR" ]; then
  render_args+=(--camera-human-ply-dir "$CAMERA_HUMAN_PLY_DIR")
  render_args+=(--camera-human-height "$CAMERA_HUMAN_HEIGHT")
  render_args+=(--camera-human-seed "$CAMERA_HUMAN_SEED")
  render_args+=(--camera-human-stride "$CAMERA_HUMAN_STRIDE")
  if [ -n "$CAMERA_HUMAN_MAX_FRAMES" ]; then
    render_args+=(--camera-human-max-frames "$CAMERA_HUMAN_MAX_FRAMES")
  fi
fi
if [ "$REVERSE_FORWARD" = "true" ]; then
  render_args+=(--reverse-forward)
fi
if [ "$NEGATE_RASTER_WORLD_XY" = "true" ]; then
  render_args+=(--negate-raster-world-xy)
fi

echo "[WAYMO] Running: ${RUNNER[*]} ${SCRIPT_DIR}/render_label_paths.py ${render_args[*]}" >&2
exec "${RUNNER[@]}" "${SCRIPT_DIR}/render_label_paths.py" "${render_args[@]}"
