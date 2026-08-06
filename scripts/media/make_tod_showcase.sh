#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_ROOT="${1:-${GS_DIR}/data2}"
BASE_NAME="${2:-0500_fpv}"
OUT_DIR="${3:-${INPUT_ROOT}/${BASE_NAME}_tod_showcase}"
OUT_VIDEO="${4:-${OUT_DIR}/${BASE_NAME}_tod_4x2.mp4}"

DURATION="${DURATION:-30}"
TILE_W="${TILE_W:-480}"
TILE_H="${TILE_H:-360}"

PRESETS=(night dawn morning noon afternoon golden_hour dusk blue_hour)

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg is required but not found in PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
selection_log="${OUT_DIR}/selection.txt"
: >"$selection_log"

selected_files=()
for preset in "${PRESETS[@]}"; do
  preset_dir="${INPUT_ROOT}/${BASE_NAME}_${preset}"
  if [ ! -d "$preset_dir" ]; then
    echo "[ERROR] Missing preset folder: ${preset_dir}" >&2
    exit 1
  fi
  mp4s=()
  while IFS= read -r mp4; do
    mp4s+=("$mp4")
  done < <(find "$preset_dir" -type f -name "*.mp4" | sort)
  if [ "${#mp4s[@]}" -eq 0 ]; then
    echo "[ERROR] No MP4s found under ${preset_dir}" >&2
    exit 1
  fi
  pick="${mp4s[$((RANDOM % ${#mp4s[@]}))]}"
  dst="${OUT_DIR}/${BASE_NAME}_${preset}.mp4"
  cp -f "$pick" "$dst"
  printf "%s\t%s\n" "$preset" "$pick" >>"$selection_log"
  selected_files+=("$dst")
done

inputs=()
filter_lines=()
for i in "${!selected_files[@]}"; do
  preset="${PRESETS[$i]}"
  label="${preset//_/ }"
  inputs+=(-stream_loop -1 -i "${selected_files[$i]}")
  line="[${i}:v]scale=${TILE_W}:${TILE_H}:force_original_aspect_ratio=decrease,"
  line+="pad=${TILE_W}:${TILE_H}:(ow-iw)/2:(oh-ih)/2,"
  line+="drawtext=text='${label}':x=10:y=10:fontsize=24:fontcolor=white:"
  line+="box=1:boxcolor=black@0.5:boxborderw=6[v${i}]"
  filter_lines+=("$line")
done

layout="0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0"
filter_complex="$(printf "%s;" "${filter_lines[@]}")"
filter_complex+=$(printf "[v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=%s[stack]" "$layout")

ffmpeg -y "${inputs[@]}" \
  -filter_complex "$filter_complex" \
  -map "[stack]" \
  -t "$DURATION" \
  -r 30 \
  -an \
  -c:v libx264 \
  -crf 20 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "$OUT_VIDEO"

echo "[DONE] Selection log: ${selection_log}"
echo "[DONE] Output video:  ${OUT_VIDEO}"
