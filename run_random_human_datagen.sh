#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-12345}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_33w.json}
SCENES_DIR=${SCENES_DIR:-./data/scenes}
TASKS_DIR=${TASKS_DIR:-./data/selected_33w}
OUTPUT_DIR=${OUTPUT_DIR:-./data/path_video_frames_random_humans_33w}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/random_human_dataset_w_ban_33w}
WORKERS=${WORKERS:-6}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-90}

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

ensure_writable_dir "$OUTPUT_DIR"

echo "Running random human data generation, seed ${SEED}..."
conda run --no-capture-output -n cuda121 python random_actor_assignments.py \
  --actor-root "${ACTOR_ROOT}" \
  --ban-list "${BAN_LIST}" \
  --assignments-out "${ASSIGNMENTS_OUT}" \
  --scenes-dir "${SCENES_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --seed "${SEED}"
echo "Assignment manifest generated at ${ASSIGNMENTS_OUT}"
conda run --no-capture-output -n cuda121 python parallel_render_paths.py \
  --assignment-manifest "${ASSIGNMENTS_OUT}" \
  --scenes-dir "${SCENES_DIR}" \
  --tasks-dir "${TASKS_DIR}" \
  --workers "${WORKERS}" \
  --minimal-frames "${MINIMAL_FRAMES}" \
  --output-dir "${OUTPUT_DIR}" \
  --render-extra-args "--overwrite --stabilize --gpu-only \
    --offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb 0.5 --show-BEV \
    --no-rgb-frames --navdp-ply-per-scene"
