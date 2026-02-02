#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
GS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
INPUT_DIR=${INPUT_DIR:-"$GS_DIR/navdata/CHINGMU_0800"}
MP4_LIST=${MP4_LIST:-"$GS_DIR/analysis/CHINGMU_0800_mp4_list.txt"}
OUTPUT_JSON=${OUTPUT_JSON:-"$GS_DIR/analysis/CHINGMU_0800_luma_report.json"}

WORKERS=${WORKERS:-1}
PROGRESS_EVERY=${PROGRESS_EVERY:-50}
SUFFIX_MODE=${SUFFIX_MODE:-luma}
SCALES=${SCALES:-"1.5 0.5 0.2"}
BASE_LUMA=${BASE_LUMA:-}
COMPUTE_BASE_LUMA=${COMPUTE_BASE_LUMA:-}
FRAME_STEP=${FRAME_STEP:-5}
PIXEL_STEP=${PIXEL_STEP:-8}
MAX_FRAMES=${MAX_FRAMES:-60}
BASE_SAMPLE_PER_SCENE=${BASE_SAMPLE_PER_SCENE:-2}
BASE_SAMPLE_SEED=${BASE_SAMPLE_SEED:-12345}
BASE_MAX_SCENES=${BASE_MAX_SCENES:-50}
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

if [[ -z "$COMPUTE_BASE_LUMA" ]]; then
  if [[ "$SUFFIX_MODE" == "luma" && -z "$BASE_LUMA" ]]; then
    COMPUTE_BASE_LUMA=1
  else
    COMPUTE_BASE_LUMA=0
  fi
fi

args=(
  "$SCRIPT_DIR/build_lighting_dataset.py"
  "$INPUT_DIR"
  --mp4-list "$MP4_LIST"
  --workers "$WORKERS"
  --progress-every "$PROGRESS_EVERY"
  --suffix-mode "$SUFFIX_MODE"
  --output-json "$OUTPUT_JSON"
)

# shellcheck disable=SC2206
scale_list=($SCALES)
args+=(--scales "${scale_list[@]}")

if [[ -n "$BASE_LUMA" ]]; then
  args+=(--base-luma "$BASE_LUMA")
elif [[ "$SUFFIX_MODE" == "luma" && "$COMPUTE_BASE_LUMA" -eq 1 ]]; then
  args+=(
    --compute-base-luma
    --frame-step "$FRAME_STEP"
    --pixel-step "$PIXEL_STEP"
    --max-frames "$MAX_FRAMES"
    --base-sample-per-scene "$BASE_SAMPLE_PER_SCENE"
    --base-sample-seed "$BASE_SAMPLE_SEED"
    --base-max-scenes "$BASE_MAX_SCENES"
  )
fi

exec "$PYTHON_BIN" "${args[@]}"
