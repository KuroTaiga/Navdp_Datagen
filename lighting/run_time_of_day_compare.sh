#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

FFMPEG_BIN=${FFMPEG_BIN:-}
if [ -n "$FFMPEG_BIN" ]; then
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
elif command -v ffmpeg >/dev/null 2>&1; then
  export IMAGEIO_FFMPEG_EXE
  IMAGEIO_FFMPEG_EXE=$(command -v ffmpeg)
fi

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" "$PYTHON_BIN")
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

TASKS_DIR=${TASKS_DIR:-"${GS_DIR}/data/interiorGS_0500_42"}
SCENES_DIR=${SCENES_DIR:-"${GS_DIR}/data/scenes"}
OUT_ROOT=${OUT_ROOT:-"${GS_DIR}/analysis/lighting_time_of_day"}
COUNT=${COUNT:-30}
SEED=${SEED:-12345}
REUSE_SAMPLE=${REUSE_SAMPLE:-true}
HEIGHT_OFFSET=${HEIGHT_OFFSET:-0.3}
VIDEO_BACKEND=${VIDEO_BACKEND:-cpu}
VIDEO_NVENC_PRESET=${VIDEO_NVENC_PRESET:-}
VIDEO_NVENC_BITRATE=${VIDEO_NVENC_BITRATE:-}

FRAME_STEP=${FRAME_STEP:-1}
PIXEL_STEP=${PIXEL_STEP:-4}

DO_BASE=${DO_BASE:-true}
DO_POST=${DO_POST:-true}
DO_RENDER=${DO_RENDER:-true}
DO_COMPARE=${DO_COMPARE:-true}

TONES=${TONES:-"golden_hour blue_hour"}

GOLDEN_STRENGTH=${GOLDEN_STRENGTH:--0.15}
GOLDEN_TEMP_K=${GOLDEN_TEMP_K:-3200}
BLUE_STRENGTH=${BLUE_STRENGTH:--0.35}
BLUE_TEMP_K=${BLUE_TEMP_K:-9000}

VIDEO_ARGS="--video-backend ${VIDEO_BACKEND}"
if [ -n "${VIDEO_NVENC_PRESET}" ]; then
  VIDEO_ARGS="${VIDEO_ARGS} --video-nvenc-preset ${VIDEO_NVENC_PRESET}"
fi
if [ -n "${VIDEO_NVENC_BITRATE}" ]; then
  VIDEO_ARGS="${VIDEO_ARGS} --video-nvenc-bitrate ${VIDEO_NVENC_BITRATE}"
fi

COMMON_RENDER_ARGS=${COMMON_RENDER_ARGS:-"--view-mode forward --gpu-only --height-offset ${HEIGHT_OFFSET} --no-rgb-frames --no-save-depth-maps --no-save-camera-metadata --no-show-BEV ${VIDEO_ARGS}"}

SAMPLE_JSON="${OUT_ROOT}/sample_${COUNT}.json"
SAMPLE_TXT="${OUT_ROOT}/sample_${COUNT}.txt"
BASE_DIR="${OUT_ROOT}/base"
BASE_METRICS="${OUT_ROOT}/metrics_base"
BASE_REPORT="${OUT_ROOT}/base_lighting.json"

mkdir -p "$OUT_ROOT"

if ! bool_true "$REUSE_SAMPLE" || [ ! -f "$SAMPLE_JSON" ]; then
  echo "[1/4] Sampling ${COUNT} paths..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/sample_paths.py" \
    --tasks-dir "$TASKS_DIR" \
    --count "$COUNT" \
    --seed "$SEED" \
    --output-json "$SAMPLE_JSON" \
    --output-txt "$SAMPLE_TXT"
else
  echo "[1/4] Reusing sample list: ${SAMPLE_JSON}"
fi

if bool_true "$DO_BASE"; then
  echo "[2/4] Base render (no lighting filter)..."
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
    --sample-json "$SAMPLE_JSON" \
    --scenes-dir "$SCENES_DIR" \
    --tasks-dir "$TASKS_DIR" \
    --output-dir "$BASE_DIR" \
    --metrics-dir "$BASE_METRICS" \
    --render-extra-args "$COMMON_RENDER_ARGS"
fi

if [ -d "$BASE_DIR" ] && [ ! -f "$BASE_REPORT" ]; then
  "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
    "$BASE_DIR" \
    --pattern "*.mp4" \
    --frame-step "$FRAME_STEP" \
    --pixel-step "$PIXEL_STEP" \
    --output-json "$BASE_REPORT"
fi

if [ ! -f "$BASE_REPORT" ]; then
  echo "[ERROR] Base lighting report missing: ${BASE_REPORT}" >&2
  echo "[ERROR] Run the base render first or set DO_BASE=true." >&2
  exit 1
fi

for tone in $TONES; do
  case "$tone" in
    golden_hour)
      TONE_STRENGTH="$GOLDEN_STRENGTH"
      TONE_TEMP_K="$GOLDEN_TEMP_K"
      ;;
    blue_hour)
      TONE_STRENGTH="$BLUE_STRENGTH"
      TONE_TEMP_K="$BLUE_TEMP_K"
      ;;
    *)
      echo "[WARN] Unknown tone '$tone' (skip)."
      continue
      ;;
  esac

  TONE_ROOT="${OUT_ROOT}/${tone}"
  POST_DIR="${TONE_ROOT}/post"
  RENDER_DIR="${TONE_ROOT}/render"
  METRICS_RENDER="${TONE_ROOT}/metrics_render"
  POST_REPORT="${TONE_ROOT}/post_report.json"
  RENDER_REPORT="${TONE_ROOT}/render_lighting.json"
  COMPARE_JSON="${TONE_ROOT}/compare.json"
  COMPARE_CSV="${TONE_ROOT}/compare.csv"

  mkdir -p "$TONE_ROOT"

  if bool_true "$DO_POST"; then
    echo "[POST] ${tone} (strength=${TONE_STRENGTH} temp_k=${TONE_TEMP_K})"
    "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/apply_light_filter_mp4.py" \
      "$BASE_DIR" \
      --pattern "*.mp4" \
      --output-dir "$POST_DIR" \
      --suffix "_${tone}" \
      --light-mode global \
      --light-strength "$TONE_STRENGTH" \
      --light-temp-k "$TONE_TEMP_K" \
      --pixel-step "$PIXEL_STEP" \
      --output-json "$POST_REPORT"
  fi

  if bool_true "$DO_RENDER"; then
    echo "[RENDER] ${tone} (strength=${TONE_STRENGTH} temp_k=${TONE_TEMP_K})"
    "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/render_sample_paths.py" \
      --sample-json "$SAMPLE_JSON" \
      --scenes-dir "$SCENES_DIR" \
      --tasks-dir "$TASKS_DIR" \
      --output-dir "$RENDER_DIR" \
      --metrics-dir "$METRICS_RENDER" \
      --render-extra-args "${COMMON_RENDER_ARGS} --light-mode global --light-strength ${TONE_STRENGTH} --light-temp-k ${TONE_TEMP_K}"
    "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_report.py" \
      "$RENDER_DIR" \
      --pattern "*.mp4" \
      --frame-step "$FRAME_STEP" \
      --pixel-step "$PIXEL_STEP" \
      --output-json "$RENDER_REPORT"
  fi

  if bool_true "$DO_COMPARE"; then
    echo "[COMPARE] ${tone}"
    COMPARE_ARGS=(--base-json "$BASE_REPORT" --output-json "$COMPARE_JSON" --output-csv "$COMPARE_CSV")
    if [ -f "$POST_REPORT" ]; then
      COMPARE_ARGS+=(--post-json "$POST_REPORT")
    fi
    if [ -f "$RENDER_REPORT" ]; then
      COMPARE_ARGS+=(--render-json "$RENDER_REPORT")
    fi
    if [ "${#COMPARE_ARGS[@]}" -le 3 ]; then
      echo "[WARN] Missing post/render reports for ${tone}; skip compare."
      continue
    fi
    "${PYTHON_CMD[@]}" "${GS_DIR}/lighting/lighting_compare_report.py" "${COMPARE_ARGS[@]}"
  fi
done

echo "Done. Outputs in ${OUT_ROOT}"
