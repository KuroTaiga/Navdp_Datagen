#!/usr/bin/env bash
set -euo pipefail

MNT=/mnt/DATA_Trans
BASE=/home/dongjk/project_files/NavDP_Jiankun_ver/navdp_api/gaussian_splatting
LOG="$BASE/TOD_lighting.log"
DEST="$MNT/navdata"
WAYMO_SRC=/home/zhangxt/workspace/drivestudio
WORKERS=${WORKERS:-3}   # deprecated for dataset-queue mode
WORKERS_WAYMO=${WORKERS_WAYMO:-1}
IMPATIENT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --impatient)
      IMPATIENT=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--impatient]"
      exit 1
      ;;
  esac
done

mkdir -p "$DEST" "$DEST/data1" "$DEST/data2" "$DEST/waymo"

items=()
items_data1=()
items_data2=()

add_item() {
  local src="$1"
  local sub="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$DEST/$sub"
    case "$sub" in
      data1/*) items_data1+=("$src|$sub") ;;
      data2/*) items_data2+=("$src|$sub") ;;
      *) items+=("$src|$sub") ;;
    esac
  else
    echo "SKIP missing: $src"
  fi
}

add_dataset() {
  local root="$1"
  local subroot="$2"
  if [[ ! -d "$root" ]]; then
    echo "SKIP missing: $root"
    return
  fi
  local base
  base=$(basename "$root")
  local sub="$subroot/$base"
  local found=0
  for d in "$root"/*; do
    if [[ -d "$d" ]]; then
      add_item "$d" "$sub"
      found=1
    fi
  done
  if (( found == 0 )); then
    add_item "$root" "$sub"
  fi
}

# data2 (completed ones commented out)
# add_dataset "$BASE/data2/0500_fpv_0.2L" "data2"
# add_dataset "$BASE/data2/0500_fpv_0.5L" "data2"
# add_dataset "$BASE/data2/0500_fpv_1.5L" "data2"
# add_dataset "$BASE/data2/33w_key1" "data2"
# add_dataset "$BASE/data2/33w_key33" "data2"
# add_dataset "$BASE/data2/33w_key65" "data2"

# data1
add_dataset "$BASE/data1/0500_42_follow_key_1" "data1"
add_dataset "$BASE/data1/33w_key2" "data1"
add_dataset "$BASE/data1/0500_fpv_npc_dense" "data1"

# waymo (dereference links)
add_item "$WAYMO_SRC" "waymo"

# merge items for dataset-queue mode
items+=( "${items_data1[@]}" "${items_data2[@]}" )

# total folders for coarse progress
total_items=${#items[@]}
DONE_FILE=/tmp/navdata_done.$$
> "$DONE_FILE"

dataset_key() {
  local label="$1"
  if [[ "$label" == "waymo" ]]; then
    echo "waymo"
    return
  fi
  local a b
  IFS=/ read -r a b _ <<< "$label"
  echo "$a/$b"
}

declare -A dataset_total
dataset_keys=()
for item in "${items[@]}"; do
  key=$(dataset_key "$item")
  dataset_total["$key"]=$(( ${dataset_total["$key"]:-0} + 1 ))
  dataset_keys+=("$key")
done
mapfile -t dataset_keys_sorted < <(printf '%s\n' "${dataset_keys[@]}" | sort -u)

STDBUF_CMD=()
if command -v stdbuf >/dev/null 2>&1; then
  STDBUF_CMD=(stdbuf -oL -eL)
fi

# rsync options: -L deref symlinks; avoid chown/perms on exFAT
RSYNC_OPTS=(-rLpt --numeric-ids --no-owner --no-group --no-perms --no-acls --no-xattrs --no-specials --no-devices --modify-window=2)
if (( WORKERS > 1 )); then
  RSYNC_OPTS+=(--info=stats2,flist0)
else
  RSYNC_OPTS+=(--info=progress2)
fi

# write per-dataset worklists (1 worker per dataset)
declare -A DATASET_LIST
LIST_FILES=()
for item in "${items[@]}"; do
  label="${item#*|}"
  key=$(dataset_key "$label")
  list="/tmp/navdata_copy_list.${key//\//_}.$$"
  if [[ -z "${DATASET_LIST[$key]:-}" ]]; then
    DATASET_LIST["$key"]="$list"
    LIST_FILES+=("$list")
  fi
  printf '%s\n' "$item" >> "$list"
done

run_queue() {
  local list_file="$1"
  local workers="$2"
  local name="${3:-queue}"
  if [[ ! -s "$list_file" ]]; then
    return 0
  fi
  echo "[QUEUE] $(date "+%F %T")  ${name} (workers=${workers})"
  xargs -P "$workers" -I{} "${STDBUF_CMD[@]}" bash -c '
    IFS="|" read -r src sub <<< "{}"
    base=$(basename "$src")
    if [[ "$sub" == "waymo" ]]; then
      label="waymo"
    else
      label="$sub/$base"
    fi
    echo "[START] $(date "+%F %T")  $label"
    rsync '"${RSYNC_OPTS[*]}"' "$src" "'"$DEST"'/$sub/"
    echo "[DONE ] $(date "+%F %T")  $label"
    echo "$label" >> "'"$DONE_FILE"'"
  ' _ {} < "$list_file" &
  echo $!
}

pids=()
for key in "${!DATASET_LIST[@]}"; do
  list="${DATASET_LIST[$key]}"
  workers=1
  if [[ "$key" == "waymo" ]]; then
    workers="$WORKERS_WAYMO"
  fi
  pid=$(run_queue "$list" "$workers" "$key") || true
  [[ -n "$pid" ]] && pids+=("$pid")
done

# print overall progress to stderr so start/done lines stay clean
progress_line() {
  local line="$1"
  if [[ -t 2 ]]; then
    printf "\r%s" "$line" >&2
  else
    printf "%s\n" "$line" >&2
  fi
}

# overall progress loop (report every 1 minute or when progress changes)
last_done=-1
last_report=0
while :; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  if (( alive == 0 )); then
    break
  fi
  done_count=0
  if [[ -f "$DONE_FILE" ]]; then
    done_count=$(wc -l < "$DONE_FILE")
  fi
  pct=0
  if (( total_items > 0 )); then
    pct=$(( done_count * 100 / total_items ))
  fi
  now=$(date +%s)
  if (( done_count != last_done )) || (( now - last_report >= 60 )); then
    progress_line "OVERALL: ${pct}% (${done_count}/${total_items} folders)"
    declare -A dataset_done
    if [[ -f "$DONE_FILE" ]]; then
      while IFS= read -r label; do
        [[ -z "$label" ]] && continue
        key=$(dataset_key "$label")
        dataset_done["$key"]=$(( ${dataset_done["$key"]:-0} + 1 ))
      done < "$DONE_FILE"
    fi
    for key in "${dataset_keys_sorted[@]}"; do
      d=${dataset_done["$key"]:-0}
      t=${dataset_total["$key"]:-0}
      progress_line "DATASET: $key ${d}/${t}"
    done
    last_done=$done_count
    last_report=$now
  fi
  sleep 10
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done
if [[ -t 2 ]]; then
  printf "\n" >&2
fi
printf "\nCopy complete.\n"

# TOD folders only after main copy finishes
tod_copy() {
  local name="$1"
  local src="$BASE/data2/$name"
  local dst="$DEST/data2/$name/"
  if (( IMPATIENT == 1 )) || ([[ -f "$LOG" ]] && grep -iE "${name}.*(done|complete|finished)" "$LOG" >/dev/null); then
    if [[ -d "$src" ]]; then
      mkdir -p "$dst"
      local found=0
      for d in "$src"/*; do
        if [[ -d "$d" ]]; then
          base=$(basename "$d")
          echo "[START] $(date "+%F %T")  data2/$name/$base"
          if (( IMPATIENT == 1 )); then
            tmp_excludes="/tmp/navdata_hues_excludes.$$.${name}.${base}"
            : > "$tmp_excludes"
            # skip all .264 in hues folders
            echo "*.264" >> "$tmp_excludes"
            rsync "${RSYNC_OPTS[@]}" --exclude-from="$tmp_excludes" "$d" "$dst"
            rm -f "$tmp_excludes"
          else
            rsync "${RSYNC_OPTS[@]}" "$d" "$dst"
          fi
          echo "[DONE ] $(date "+%F %T")  data2/$name/$base"
          found=1
        fi
      done
      if (( found == 0 )); then
        echo "[START] $(date "+%F %T")  data2/$name"
        if (( IMPATIENT == 1 )); then
          tmp_excludes="/tmp/navdata_hues_excludes.$$.${name}"
          : > "$tmp_excludes"
          echo "*.264" >> "$tmp_excludes"
          rsync "${RSYNC_OPTS[@]}" --exclude-from="$tmp_excludes" "$src" "$dst"
          rm -f "$tmp_excludes"
        else
          rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
        fi
        echo "[DONE ] $(date "+%F %T")  data2/$name"
      fi
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

rm -f "${LIST_FILES[@]}" "$DONE_FILE"
