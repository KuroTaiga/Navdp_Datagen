#!/usr/bin/env bash
set -euo pipefail

MNT=/mnt/DATA_Trans
BASE=/home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
LOG="$BASE/TOD_lighting.log"
DEST="$MNT/navdata"
WAYMO_SRC=/home/zhangxt/workspace/drivestudio
WORKERS=${WORKERS:-3}   # set WORKERS=1 to avoid parallel

mkdir -p "$DEST" "$DEST/data1" "$DEST/data2" "$DEST/waymo"

items=()

add_item() {
  local src="$1"
  local sub="$2"
  if [[ -d "$src" ]]; then
    items+=("$src|$sub")
  else
    echo "SKIP missing: $src"
  fi
}

# data2
add_item "$BASE/data2/0500_fpv_0.2L" "data2"
add_item "$BASE/data2/0500_fpv_0.5L" "data2"
add_item "$BASE/data2/0500_fpv_1.5L" "data2"
add_item "$BASE/data2/33w_key1" "data2"
add_item "$BASE/data2/33w_key33" "data2"
add_item "$BASE/data2/33w_key65" "data2"

# data1
add_item "$BASE/data1/0500_42_follow_key_1" "data1"
add_item "$BASE/data1/33w_key2" "data1"
add_item "$BASE/data1/0500_fpv_npc_dense" "data1"

# waymo (dereference links)
add_item "$WAYMO_SRC" "waymo"

# total folders for coarse progress
total_items=${#items[@]}
DONE_FILE=/tmp/navdata_done.$$
> "$DONE_FILE"

# rsync options: -L deref symlinks
RSYNC_OPTS=(-aHAXL --numeric-ids)
if (( WORKERS > 1 )); then
  RSYNC_OPTS+=(--info=stats2,flist0)
else
  RSYNC_OPTS+=(--info=progress2)
fi

# write worklist
LIST=/tmp/navdata_copy_list.$$
printf '%s\n' "${items[@]}" > "$LIST"

# run copies in parallel
xargs -P "$WORKERS" -n 1 -I{} bash -c '
  IFS="|" read -r src sub <<< "{}"
  echo "[START] $(date "+%F %T")  $src -> '"$DEST"'/$sub/"
  rsync '"${RSYNC_OPTS[*]}"' "$src" "'"$DEST"'/$sub/"
  echo "[DONE ] $(date "+%F %T")  $src -> '"$DEST"'/$sub/"
  echo "$src" >> "'"$DONE_FILE"'"
' _ {} < "$LIST" &
copy_pid=$!

# print overall progress to stderr so start/done lines stay clean
progress_line() {
  local line="$1"
  if [[ -t 2 ]]; then
    printf "\r%s" "$line" >&2
  else
    printf "%s\n" "$line" >&2
  fi
}

# overall progress loop
while kill -0 "$copy_pid" 2>/dev/null; do
  done_count=0
  if [[ -f "$DONE_FILE" ]]; then
    done_count=$(wc -l < "$DONE_FILE")
  fi
  pct=0
  if (( total_items > 0 )); then
    pct=$(( done_count * 100 / total_items ))
  fi
  progress_line "OVERALL: ${pct}% (${done_count}/${total_items} folders)"
  sleep 10
done
wait "$copy_pid"
if [[ -t 2 ]]; then
  printf "\n" >&2
fi
printf "\nCopy complete.\n"

# TOD folders only after main copy finishes
tod_copy() {
  local name="$1"
  local src="$BASE/data2/$name"
  local dst="$DEST/data2/"
  if [[ -f "$LOG" ]] && grep -iE "${name}.*(done|complete|finished)" "$LOG" >/dev/null; then
    if [[ -d "$src" ]]; then
      echo "[START] $(date "+%F %T")  $src -> $dst"
      rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
      echo "[DONE ] $(date "+%F %T")  $src -> $dst"
    else
      echo "SKIP missing: $src"
    fi
  else
    echo "TOD not done per log: $name"
  fi
}

for d in \
  0500_fpv_afternoon \
  0500_fpv_blue_hour \
  0500_fpv_dawn \
  0500_fpv_dusk \
  0500_fpv_golden_hour \
  0500_fpv_morning \
  0500_fpv_night \
  0500_fpv_noon
do
  tod_copy "$d"
done

rm -f "$LIST" "$DONE_FILE"
