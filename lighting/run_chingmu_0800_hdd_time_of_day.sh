#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
GS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
INPUT_DIR=${INPUT_DIR:-"$GS_DIR/navdata/CHINGMU_0800"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$GS_DIR/navdata"}
MP4_LIST=${MP4_LIST:-"$GS_DIR/analysis/CHINGMU_0800_mp4_list.txt"}
OUTPUT_JSON=${OUTPUT_JSON:-"$GS_DIR/analysis/CHINGMU_0800_tod_report.json"}

WORKERS=${WORKERS:-1}
PROGRESS_EVERY=${PROGRESS_EVERY:-50}
COMPUTE_BACKEND=${COMPUTE_BACKEND:-cpu}
VIDEO_BACKEND=${VIDEO_BACKEND:-cpu}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}
PRESETS=${PRESETS:-}
MAX_FILES=${MAX_FILES:-}
MAX_FRAMES=${MAX_FRAMES:-}
RESUME=${RESUME:-1}
REFRESH_LIST=${REFRESH_LIST:-0}

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Input not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$MP4_LIST" || "$REFRESH_LIST" -eq 1 ]]; then
  mkdir -p "$(dirname "$MP4_LIST")"
  echo "[LIST] building mp4 list for $INPUT_DIR..."
  LC_ALL=C find "$INPUT_DIR" -type f -name "*.mp4" -print | LC_ALL=C sort > "$MP4_LIST"
fi

args=(
  "$SCRIPT_DIR/build_time_of_day_dataset.py"
  "$INPUT_DIR"
  --mp4-list "$MP4_LIST"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --progress-every "$PROGRESS_EVERY"
  --compute-backend "$COMPUTE_BACKEND"
  --video-backend "$VIDEO_BACKEND"
  --output-json "$OUTPUT_JSON"
)

if [[ -n "$PRESETS" ]]; then
  # shellcheck disable=SC2206
  args+=(--presets $PRESETS)
fi
if [[ -n "$VIDEO_NVENC_PRESET" ]]; then
  args+=(--video-nvenc-preset "$VIDEO_NVENC_PRESET")
fi
if [[ -n "$VIDEO_NVENC_BITRATE" ]]; then
  args+=(--video-nvenc-bitrate "$VIDEO_NVENC_BITRATE")
fi
if [[ -n "$MAX_FILES" ]]; then
  args+=(--max-files "$MAX_FILES")
fi
if [[ -n "$MAX_FRAMES" ]]; then
  args+=(--max-frames "$MAX_FRAMES")
fi
if [[ "$RESUME" -eq 0 ]]; then
  args+=(--no-resume)
fi

exec "$PYTHON_BIN" "${args[@]}"
