#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_ROOT=${SRC_ROOT:-}
if [ -z "$SRC_ROOT" ]; then
  echo "[ERROR] SRC_ROOT is required. Set it to the source human-sequence export root." >&2
  exit 1
fi
MESHS_ROOT="${SRC_ROOT%/}/meshes"
DST_ROOT=${DST_ROOT:-${REPO_ROOT}/data/SHHQ_gs/walking}
DRY_RUN=${DRY_RUN:-false}

bool_true() {
  local val
  val=$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')
  case "$val" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

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
    find "$MESHS_ROOT" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9][0-9]*' \
      | while IFS= read -r uid_dir; do basename "$uid_dir"; done \
      | sort
  fi
}

UID_LIST=()
while IFS= read -r uid; do
  UID_LIST+=("$uid")
done < <(collect_uids "$@")
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
  if bool_true "$DRY_RUN"; then
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
