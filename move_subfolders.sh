#!/bin/bash
SRC="./data/path_video_frames_random_humans_33w"
DEST="lenovo@192.168.151.40:/mnt/DATA/navdp_data_33w"
JOBS=12    # how many subfolders to copy in parallel

cd "$SRC" || exit 1

export DEST   # so the subshells see it

find . -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -n1 -P"$JOBS" bash -c '
      d="$1"
      name="${d#./}"

      echo ">>> [$name] Moving subfolder"

      rsync -avh --partial --inplace --remove-source-files \
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
