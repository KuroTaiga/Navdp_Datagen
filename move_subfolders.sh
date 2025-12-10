#!/bin/bash
SRC="./data/path_video_frames_random_humans_33w"
DEST="lenovo@192.168.151.40:/mnt/DATA/navdp_data_33w"
JOBS=12    # how many subfolders to copy in parallel
LAST_FOLDER_NUMBER=400          # only move folders whose numeric prefix is below this
OVERWRITE_REMOTE=${OVERWRITE_REMOTE:-1}  # set to 1 to force overwrite on rsync

cd "$SRC" || exit 1

export DEST OVERWRITE_REMOTE   # so the subshells see it

mapfile -d '' SUBFOLDERS < <(find . -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

PRIORITIZED=()
SKIPPED=()

for entry in "${SUBFOLDERS[@]}"; do
    name="${entry#./}"
    [[ -z "$name" ]] && continue

    if [[ $name =~ ([0-9]+) ]]; then
        folder_num=$((10#${BASH_REMATCH[1]}))
    else
        folder_num=-1
    fi

    if (( folder_num >= 0 && folder_num < LAST_FOLDER_NUMBER )); then
        PRIORITIZED+=("$name")
    else
        SKIPPED+=("$name")
    fi
done

if [ ${#PRIORITIZED[@]} -eq 0 ]; then
    echo "No folders found below $LAST_FOLDER_NUMBER in $SRC"
    exit 0
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "Skipping ${#SKIPPED[@]} folders with id >= $LAST_FOLDER_NUMBER"
fi

printf '%s\0' "${PRIORITIZED[@]}" \
  | xargs -0 -n1 -P"$JOBS" bash -c '
      d="$1"
      name="${d#./}"

      echo ">>> [$name] Moving subfolder"

      overwrite_flag="${OVERWRITE_REMOTE:-0}"
      rsync_extra=""
      if [ "$overwrite_flag" = "1" ]; then
          rsync_extra="--ignore-times"
      fi

      rsync -avh --partial --inplace --remove-source-files $rsync_extra \
          "./$name/" \
          "$DEST/$name/"

      rc=$?
      if [ $rc -ne 0 ]; then
          echo "!!! [$name] rsync failed (exit $rc), leaving folder in place"
          # leave folder as-is; you can rerun script later
          exit 0
      fi

      # Clean up empty dirs under this subfolder
      find "./$name" -type d -empty -delete

      # Try remove top dir if empty
      rmdir "./$name" 2>/dev/null && echo "    Removed local $name"
  ' _
