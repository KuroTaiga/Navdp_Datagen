#!/bin/bash
set -euo pipefail

MESHS_ROOT="/mnt/nas/jiankundong/SHHQ_walk_fbx/gs/meshs"
DST="./data/human_gs_source"

echo "[INFO] MESHS_ROOT = $MESHS_ROOT"
echo "[INFO] DST        = $DST"
echo "[INFO] Scanning UID folders under MESHS_ROOT..."

# Collect UID dirs (numeric names directly under MESHS_ROOT)
mapfile -t UID_DIRS < <(find "$MESHS_ROOT" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9][0-9]*' | sort)

echo "[INFO] Found ${#UID_DIRS[@]} UID folders."
for d in "${UID_DIRS[@]}"; do
    echo "  UID DIR: $d"
done

if (( ${#UID_DIRS[@]} == 0 )); then
    echo "[ERROR] No UID folders found. Check MESHS_ROOT and folder naming."
    exit 1
fi

for UID_DIR in "${UID_DIRS[@]}"; do
    UID_BASENAME=$(basename "$UID_DIR")

    MOTION_DIR="$UID_DIR/motion_seq_cleaned/${UID_BASENAME}_motion"
    TARGET="$DST/$UID_BASENAME"

    echo
    echo "==== UID: $UID_BASENAME ===="
    echo "  UID_DIR    : $UID_DIR"
    echo "  MOTION_DIR : $MOTION_DIR"
    echo "  TARGET     : $TARGET"

    if [ ! -d "$MOTION_DIR" ]; then
        echo "  !! Skipping: motion dir not found"
        continue
    fi

    mkdir -p "$TARGET"

    echo "  scanning .ply files..."
    FILES=$(find "$MOTION_DIR" -maxdepth 1 -type f -name "*.ply" | sort)

    if [ -z "$FILES" ]; then
        echo "  !! No .ply files found in $MOTION_DIR"
        continue
    fi

    for f in $FILES; do
        b=$(basename "$f")
        echo "  copying: $b"
        rsync -ah --info=progress2 --inplace "$f" "$TARGET/$b"
        echo "  done: $b"
    done

    echo "---- done UID: $UID_BASENAME ----"
done

echo
echo "[INFO] All done."
