#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

# Ensure we have a Python interpreter available (needed for path resolution helper below).

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "[ERROR] python3 is required but was not found in PATH." >&2
    exit 1
  fi
fi

# Tiny helper for consistent usage errors so RESUME mode is easy to discover.
show_usage_and_exit() {
  echo "Usage: $(basename "$0") [RESUME <log-file>]" >&2
  exit 1
}

# CLI parsing: optional leading "RESUME <log>" pair switches to resume mode and
# consumes the following logfile argument so the remainder of the script can lean
# on env vars only.
RESUME_MODE=false
RESUME_LOG_PATH=""
if [ $# -gt 0 ]; then
  if [ "$1" = "RESUME" ]; then
    RESUME_MODE=true
    shift
    if [ $# -lt 1 ]; then
      echo "[RESUME] ERROR: log file path missing." >&2
      show_usage_and_exit
    fi
    RESUME_LOG_PATH="$1"
    shift
  else
    echo "[ERROR] Unknown argument: $1" >&2
    show_usage_and_exit
  fi
fi
if [ $# -gt 0 ]; then
  echo "[ERROR] Unexpected arguments: $*" >&2
  show_usage_and_exit
fi

# Convenience wrapper so we can expand relative paths and keep the script POSIX-ish.
abspath() {
  "$PYTHON_BIN" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

# Storage toggles (set env vars to true/false/yes/no) so the same runner can ship
# data to different targets (local/NAS/remote SSH).
ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE:-true}
ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE:-false}
ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE:-true}
CLEAR_LOCAL_OUTPUT_DIR=${CLEAR_LOCAL_OUTPUT_DIR:-true}

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
  && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
  && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
  exit 1
fi

# Core configuration for assignment planning + rendering. Most callers just tweak
# DATA roots or seeds via environment variables.
SEED=${SEED:-12345}
CONDA_ENV=${CONDA_ENV:-cuda121}
ACTOR_ROOT=${ACTOR_ROOT:-./data/human_gs_source}
BAN_LIST=${BAN_LIST:-${ACTOR_ROOT}/BanList.txt}
ASSIGNMENTS_OUT=${ASSIGNMENTS_OUT:-./data/actor_assignments_w_ban_65k.json}
SCENES_DIR=${SCENES_DIR:-./data/scenes}
TASKS_DIR=${TASKS_DIR:-./data/selected_65k}
OUTPUT_DIR=${OUTPUT_DIR:-./data/path_video_frames_random_humans_65k}
OFFLOAD_NAS_DIR=${OFFLOAD_NAS_DIR:-/mnt/nas/jiankundong/random_human_dataset_w_ban_65k}
OFFLOAD_MIN_FREE_GB=${OFFLOAD_MIN_FREE_GB:-0.5}
REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/srv/navdp}}
REMOTE_SSH_TARGET=${REMOTE_SSH_TARGET:-user@other-training-pc}
LOCAL_OUTPUT_BASENAME="$(basename "$OUTPUT_DIR")"
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${LOCAL_OUTPUT_BASENAME}"
WORKERS=${WORKERS:-6}
MINIMAL_FRAMES=${MINIMAL_FRAMES:-38}
RESERVE_VRAM_GB=${RESERVE_VRAM_GB:-4}
if ! [[ "$RESERVE_VRAM_GB" =~ ^[0-9]+$ ]]; then
  echo "[VRAM] ERROR: RESERVE_VRAM_GB must be an integer value (received '$RESERVE_VRAM_GB')." >&2
  exit 1
fi

# VRAM throttling: spawn a small Python helper that grabs a configurable chunk of
# GPU memory so other users do not over-subscribe the card while this run is in
# flight. The tensor holder stays alive until we exit (trap below).
VRAM_RESERVATION_PID=""
reserve_vram() {
  local reserve_gb="$1"
  if [ -z "$reserve_gb" ]; then
    return
  fi
  local bytes=$((reserve_gb * 1024 * 1024 * 1024))
  if [ "$bytes" -le 0 ]; then
    return
  fi
  echo "[VRAM] Reserving approximately ${reserve_gb} GiB of GPU memory (set RESERVE_VRAM_GB=0 to disable)."
  RESERVE_VRAM_BYTES="$bytes" conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY' &
import os
import sys
import time

try:
    import torch
except Exception as exc:  # pylint: disable=broad-except
    print(f"[VRAM] ERROR: Unable to import torch: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)

target_bytes = int(os.environ.get("RESERVE_VRAM_BYTES", "0"))
if target_bytes <= 0:
    sys.exit(0)

device = torch.device("cuda")
tensors = []
float_elems = target_bytes // 4
if float_elems:
    tensors.append(torch.empty((float_elems,), dtype=torch.float32, device=device))
remainder = target_bytes % 4
if remainder:
    tensors.append(torch.empty((remainder,), dtype=torch.uint8, device=device))

reserved = sum(t.element_size() * t.numel() for t in tensors)
dev_index = torch.cuda.current_device()
dev_name = torch.cuda.get_device_name(dev_index)
print(f"[VRAM] Reserved {reserved / (1024 ** 3):.2f} GiB on cuda:{dev_index} ({dev_name}).", flush=True)
try:
    while True:
        time.sleep(30)
except KeyboardInterrupt:
    pass
PY
  VRAM_RESERVATION_PID=$!
  sleep 1
  if ! kill -0 "$VRAM_RESERVATION_PID" >/dev/null 2>&1; then
    echo "[VRAM] ERROR: Failed to start reservation helper." >&2
    exit 1
  fi
}

release_vram() {
  if [ -n "$VRAM_RESERVATION_PID" ]; then
    kill "$VRAM_RESERVATION_PID" >/dev/null 2>&1 || true
    wait "$VRAM_RESERVATION_PID" >/dev/null 2>&1 || true
    echo "[VRAM] Released reserved GPU memory."
    VRAM_RESERVATION_PID=""
  fi
}

# Always tear down the VRAM hog helper even on errors.
trap release_vram EXIT

RESUME_LOG_ABS=""
# Resume bookkeeping: we reuse the assignment manifest referenced inside the
# provided log file and disable destructive cleanup so partially-generated data is
# preserved.
if $RESUME_MODE; then
  RESUME_LOG_ABS=$(abspath "$RESUME_LOG_PATH")
  if [ ! -f "$RESUME_LOG_ABS" ]; then
    echo "[RESUME] ERROR: Log file not found: $RESUME_LOG_PATH" >&2
    exit 1
  fi
  manifest_line=$(grep -m1 "Assignment manifest generated at" "$RESUME_LOG_ABS" || true)
  if [ -z "$manifest_line" ]; then
    echo "[RESUME] ERROR: Could not locate assignment manifest line in $RESUME_LOG_PATH" >&2
    exit 1
  fi
  resume_manifest_path="${manifest_line#*Assignment manifest generated at }"
  resume_manifest_path="${resume_manifest_path%% *}"
  if [ -z "$resume_manifest_path" ]; then
    echo "[RESUME] ERROR: Failed to parse assignment manifest path from log." >&2
    exit 1
  fi
  if [[ "$resume_manifest_path" != /* ]]; then
    resume_manifest_path="${resume_manifest_path#./}"
    resume_manifest_path="${SCRIPT_DIR}/${resume_manifest_path}"
  fi
  ASSIGNMENTS_OUT=$(abspath "$resume_manifest_path")
  if [ ! -f "$ASSIGNMENTS_OUT" ]; then
    echo "[RESUME] ERROR: Assignment manifest missing: $ASSIGNMENTS_OUT" >&2
    exit 1
  fi
  echo "[RESUME] Using manifest $ASSIGNMENTS_OUT (derived from $RESUME_LOG_PATH)"
  CLEAR_LOCAL_OUTPUT_DIR=false
fi

# Utility: guarantee output dir exists + is writable before we drop a ton of
# frames in there.
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

# Optionally wipe previous contents to keep runs deterministic unless resume
# mode disabled the cleanup step earlier.
prepare_local_output_dir() {
  local target="$1"
  ensure_writable_dir "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

prepare_local_output_dir "$OUTPUT_DIR"

# Connectivity sanity check so we fail early if the NAS is unreachable before any
# heavy compute starts.
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  NAS_TEST_DIR="${OFFLOAD_NAS_DIR}/__connectivity_check__"
  if mkdir -p "${NAS_TEST_DIR}" \
    && : > "${NAS_TEST_DIR}/.touch" \
    && rm -f "${NAS_TEST_DIR}/.touch"; then
    echo "[CHECK] NAS reachable at ${OFFLOAD_NAS_DIR}"
  else
    echo "[CHECK] ERROR: cannot write to NAS ${OFFLOAD_NAS_DIR}" >&2
    exit 1
  fi
fi

echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"

if [ "$RESERVE_VRAM_GB" -gt 0 ]; then
  reserve_vram "$RESERVE_VRAM_GB"
fi

# Assignment planning is deterministic: in resume mode we skip generation and
# reuse the previous manifest so scene/actor pairings stay stable.
if $RESUME_MODE; then
  echo "[RESUME] Skipping assignment generation and reusing ${ASSIGNMENTS_OUT}"
else
  echo "[RUN] Random human data generation, seed ${SEED}"
  conda run --no-capture-output -n "$CONDA_ENV" python random_actor_assignments.py \
    --actor-root "${ACTOR_ROOT}" \
    --ban-list "${BAN_LIST}" \
    --assignments-out "${ASSIGNMENTS_OUT}" \
    --scenes-dir "${SCENES_DIR}" \
    --tasks-dir "${TASKS_DIR}" \
    --seed "${SEED}"
  echo "Assignment manifest generated at ${ASSIGNMENTS_OUT}"
fi

render_extra_snippets=(
  "--overwrite --stabilize --gpu-only --show-BEV --no-rgb-frames --navdp-ply-per-scene"
)
# Rendering CLI snippets are composed here so storage flags can extend/override
# behavior (NAS uploads, BEV toggles, etc.) without duplicating the Python call.
if storage_bool_true "$ENABLE_NAS_STORAGE"; then
  render_extra_snippets+=("--offload-nas-dir ${OFFLOAD_NAS_DIR} --offload-min-free-gb ${OFFLOAD_MIN_FREE_GB}")
fi

parallel_cmd=(
  conda run --no-capture-output -n "$CONDA_ENV" python parallel_render_paths.py
  --assignment-manifest "${ASSIGNMENTS_OUT}"
  --scenes-dir "${SCENES_DIR}"
  --tasks-dir "${TASKS_DIR}"
  --workers "${WORKERS}"
  --minimal-frames "${MINIMAL_FRAMES}"
  --output-dir "${OUTPUT_DIR}"
)
# Thread the resume log into the renderer so it can skip completed scene/actor
# pairs. Remaining CLI snippets (overwrite/offload/etc.) are appended below.
if [ -n "$RESUME_LOG_ABS" ]; then
  parallel_cmd+=(--skip-completed-log "$RESUME_LOG_ABS")
fi
for snippet in "${render_extra_snippets[@]}"; do
  parallel_cmd+=(--render-extra-args "$snippet")
done

render_status=0
set +e
"${parallel_cmd[@]}"
render_status=$?
set -e
if [ $render_status -ne 0 ]; then
  echo "[WARN] parallel_render_paths.py exited with status ${render_status}, continuing per request."
fi

if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
  # Offload finished results to the remote training PC via rsync/SSH when
  # enabled. Keeps the local disk tidy while ensuring central storage has copies.
  storage_sync_remote "$OUTPUT_DIR" "$REMOTE_TARGET_DIR"
fi

if ! storage_bool_true "$ENABLE_LOCAL_STORAGE"; then
  # When purely offloading to NAS/remote, purge local outputs to conserve disk.
  if [ -d "$OUTPUT_DIR" ]; then
    echo "[STORAGE] Removing local outputs at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
  fi
fi

exit $render_status
