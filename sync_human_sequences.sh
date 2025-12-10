#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_ROOT=${SRC_ROOT:-/mnt/nas/jiankundong/SHHQ_walk_fbx_new}
MESHS_ROOT="${SRC_ROOT%/}/meshes"
DST_ROOT=${DST_ROOT:-${SCRIPT_DIR}/data/human_gs_source}
DRY_RUN=${DRY_RUN:-false}

echo "[CONFIG] SRC_ROOT=${SRC_ROOT}"
echo "[CONFIG] MESHS_ROOT=${MESHS_ROOT}"
echo "[CONFIG] DST_ROOT=${DST_ROOT}"
echo "[CONFIG] DRY_RUN=${DRY_RUN}"

if [ ! -d "$MESHS_ROOT" ]; then
  echo "[ERROR] Mesh root not found: ${MESHS_ROOT}" >&2
  exit 1
fi
mkdir -p "$DST_ROOT"

collect_uids() {
  if [ "$#" -gt 0 ]; then
    printf '%s\n' "$@"
  else
    find "$MESHS_ROOT" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9][0-9]*' -printf '%f\n' | sort
  fi
}

mapfile -t UID_LIST < <(collect_uids "$@")
echo "[INFO] Found ${#UID_LIST[@]} UID folders to sync."

sync_one_uid() {
  local uid="$1"
  local motion_dir="${MESHS_ROOT}/${uid}/motion_seq_cleaned/${uid}_motion"
  local dst_dir="${DST_ROOT}/${uid}"

  echo
  echo "==== UID ${uid} ===="
  echo "  motion_dir: ${motion_dir}"
  echo "  dst_dir   : ${dst_dir}"

  if [ ! -d "$motion_dir" ]; then
    echo "  !! Skipping: motion dir missing."
    return
  fi

  mkdir -p "$dst_dir"
  local rsync_args=(-ah --info=progress2)
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    rsync_args+=(--dry-run)
  fi

  rsync "${rsync_args[@]}" \
    --include "frame_*.ply" \
    --include "*.json" \
    --include "*.txt" \
    --exclude "*" \
    "${motion_dir}/" "${dst_dir}/"
}

for uid in "${UID_LIST[@]}"; do
  sync_one_uid "$uid"
done

echo
echo "[DONE] Synced ${#UID_LIST[@]} UID directories into ${DST_ROOT}."
