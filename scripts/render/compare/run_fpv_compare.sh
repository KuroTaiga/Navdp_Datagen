#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_fpv_compare.sh <SCENE_PREFIX> <LABEL_ID|auto> <OUTPUT_ROOT> [SCENES_DIR] [TASKS_DIR]
#
# Example:
#   scripts/run_fpv_compare.sh 0001_ auto ./tmp_znear_test
#
# Notes:
# - SCENE_PREFIX can be partial (e.g., "0001_") and will match the first
#   directory under SCENES_DIR that starts with the prefix (sorted).
# - LABEL_ID should match the label JSON stem (without .json).
#   Use "auto" (or "-") to pick the first label JSON in the scene task folder.

SCENE_PREFIX="${1:-}"
LABEL_ID="${2:-}"
OUTPUT_ROOT="${3:-}"
SCENES_DIR="${4:-./data/CHINGMU_scenes_rescaled}"
TASKS_DIR="${5:-./data/CHINGMU_75_rescaled_0800_42_iter1}"

if [ -z "$SCENE_PREFIX" ] || [ -z "$OUTPUT_ROOT" ]; then
  echo "Usage: $0 <SCENE_PREFIX> <LABEL_ID|auto> <OUTPUT_ROOT> [SCENES_DIR] [TASKS_DIR]" >&2
  exit 1
fi

SCENE_ID="$(
  python3 - "$SCENE_PREFIX" "$SCENES_DIR" <<'PY'
import os
import sys
from pathlib import Path

scene_prefix = sys.argv[1]
scenes_dir = Path(sys.argv[2]).expanduser().resolve()
if not scenes_dir.is_dir():
    raise SystemExit(f"[ERROR] scenes dir not found: {scenes_dir}")

matches = sorted(
    p.name for p in scenes_dir.iterdir()
    if p.is_dir() and p.name.startswith(scene_prefix)
)
if not matches:
    raise SystemExit(f"[ERROR] no scene dirs matching prefix '{scene_prefix}' in {scenes_dir}")

# Pick first match; print others as info.
print(matches[0])
if len(matches) > 1:
    print("[INFO] Multiple matches found; using first:", matches[0], file=sys.stderr)
    for m in matches[1:5]:
        print("[INFO]   also:", m, file=sys.stderr)
PY
)"

if [ -z "${LABEL_ID}" ] || [ "${LABEL_ID}" = "auto" ] || [ "${LABEL_ID}" = "-" ]; then
  LABEL_ID="$(
    python3 - "$SCENE_ID" "$TASKS_DIR" <<'PY'
import sys
from pathlib import Path

scene_id = sys.argv[1]
tasks_dir = Path(sys.argv[2]).expanduser().resolve()
scene_task_dir = tasks_dir / scene_id

label_dir = scene_task_dir / "label_paths"
if not label_dir.is_dir():
    label_dir = scene_task_dir

if not label_dir.is_dir():
    raise SystemExit(f"[ERROR] task dir not found: {label_dir}")

jsons = sorted(
    p for p in label_dir.glob("*.json")
    if not p.name.endswith("_detailed.json") and p.name != "summary.json"
)
if not jsons:
    raise SystemExit(f"[ERROR] no label json files in {label_dir}")

print(jsons[0].stem)
if len(jsons) > 1:
    print("[INFO] Multiple labels found; using first:", jsons[0].stem, file=sys.stderr)
    for p in jsons[1:5]:
        print("[INFO]   also:", p.stem, file=sys.stderr)
PY
  )"
fi

echo "[INFO] Using scene: ${SCENE_ID}"
echo "[INFO] Using label: ${LABEL_ID}"
echo "[INFO] Scenes dir: ${SCENES_DIR}"
echo "[INFO] Tasks dir:  ${TASKS_DIR}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

python3 render_label_paths.py \
  --scenes-dir "${SCENES_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --scene "${SCENE_ID}" \
  --label-id "${LABEL_ID}" \
  --output-dir "${OUTPUT_ROOT}/default" \
  --rgb-frames --no-video \
  --video-backend cpu \
  --no-save-depth-maps --no-save-camera-metadata --no-save-follow-metadata

python3 render_label_paths.py \
  --scenes-dir "${SCENES_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --scene "${SCENE_ID}" \
  --label-id "${LABEL_ID}" \
  --output-dir "${OUTPUT_ROOT}/zn0p1_zf100" \
  --rgb-frames --no-video \
  --video-backend cpu \
  --znear 0.1 --zfar 100 \
  --no-save-depth-maps --no-save-camera-metadata --no-save-follow-metadata
