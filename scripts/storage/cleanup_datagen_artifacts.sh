#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/storage/cleanup_datagen_artifacts.sh [--dry-run] ROOT [...]

Deletes:
  - *.plt files
  - scene/path-level *.ply files
  - per-path *_BEV.png files

Pass roots as arguments or set CLEANUP_ROOTS as a space-separated list.
Use --dry-run to only print what would be deleted.
EOF
  exit 0
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [ "$#" -gt 0 ]; then
  targets=("$@")
elif [ -n "${CLEANUP_ROOTS:-}" ]; then
  # shellcheck disable=SC2206
  targets=(${CLEANUP_ROOTS})
else
  echo "[ERROR] Provide at least one dataset root or set CLEANUP_ROOTS." >&2
  exit 1
fi

for t in "${targets[@]}"; do
  if [[ -d "$t" ]]; then
    echo "Removing scene/path-level .ply files under: $t"
    while IFS= read -r -d '' file; do
      echo "[DEL] $file"
      if [ "$DRY_RUN" = false ]; then
        rm -f "$file"
      fi
    done < <(find "$t" -mindepth 3 -maxdepth 4 -type f -name "*.ply" -print0)

    echo "Removing per-path BEV PNGs under: $t"
    while IFS= read -r -d '' file; do
      echo "[DEL] $file"
      if [ "$DRY_RUN" = false ]; then
        rm -f "$file"
      fi
    done < <(find "$t" -mindepth 3 -maxdepth 4 -type f -name "*_BEV.png" -print0)
  else
    echo "Skip missing dir: $t"
  fi
done
