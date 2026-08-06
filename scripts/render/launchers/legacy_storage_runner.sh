#!/usr/bin/env bash

# Shared process/storage helpers for the root legacy datagen launchers. Callers
# own render defaults and command construction; this file only manages storage
# lifecycle and interruption behavior through the existing global variables.

navdp_init_storage_mode() {
  if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
    && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
    && ! storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    echo "[STORAGE] ERROR: At least one of ENABLE_LOCAL_STORAGE, ENABLE_NAS_STORAGE, ENABLE_REMOTE_STORAGE must be true." >&2
    exit 1
  fi

  REMOTE_ONLY_STORAGE=false
  if ! storage_bool_true "$ENABLE_LOCAL_STORAGE" \
    && ! storage_bool_true "$ENABLE_NAS_STORAGE" \
    && storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    REMOTE_ONLY_STORAGE=true
  fi

  : "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=false}"
  if [ "$REMOTE_ONLY_STORAGE" = true ]; then
    : "${REMOTE_SYNC_REMOVE_SOURCE_FILES:=true}"
  fi

  ENABLE_REMOTE_STORAGE_GUARD=${ENABLE_REMOTE_STORAGE_GUARD:-true}
  REMOTE_STORAGE_LIMIT_GB=${REMOTE_STORAGE_LIMIT_GB:-500}
  REMOTE_STORAGE_RESUME_GB=${REMOTE_STORAGE_RESUME_GB:-200}
  REMOTE_STORAGE_GUARD_INTERVAL_SECS=${REMOTE_STORAGE_GUARD_INTERVAL_SECS:-60}
  REMOTE_STORAGE_GUARD_ENABLED=false
  if [ "$REMOTE_ONLY_STORAGE" = true ] && storage_bool_true "$ENABLE_REMOTE_STORAGE_GUARD"; then
    REMOTE_STORAGE_GUARD_ENABLED=true
  fi

  SETSID_BIN=""
  if command -v setsid >/dev/null 2>&1; then
    SETSID_BIN=$(command -v setsid)
  fi
  if [ "$REMOTE_STORAGE_GUARD_ENABLED" = true ] && [ -z "$SETSID_BIN" ]; then
    echo "[STORAGE] WARN: setsid is unavailable; disabling remote storage guard throttling." >&2
    REMOTE_STORAGE_GUARD_ENABLED=false
  fi
}

navdp_validate_storage_guard_settings() {
  if ! [[ "$REMOTE_STORAGE_LIMIT_GB" =~ ^[0-9]+$ ]]; then
    echo "[STORAGE] ERROR: REMOTE_STORAGE_LIMIT_GB must be an integer value (received '$REMOTE_STORAGE_LIMIT_GB')." >&2
    exit 1
  fi
  if ! [[ "$REMOTE_STORAGE_RESUME_GB" =~ ^[0-9]+$ ]]; then
    echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB must be an integer value (received '$REMOTE_STORAGE_RESUME_GB')." >&2
    exit 1
  fi
  if [ "$REMOTE_STORAGE_RESUME_GB" -ge "$REMOTE_STORAGE_LIMIT_GB" ]; then
    echo "[STORAGE] ERROR: REMOTE_STORAGE_RESUME_GB (${REMOTE_STORAGE_RESUME_GB}GB) must be less than REMOTE_STORAGE_LIMIT_GB (${REMOTE_STORAGE_LIMIT_GB}GB)." >&2
    exit 1
  fi
  if ! [[ "$REMOTE_STORAGE_GUARD_INTERVAL_SECS" =~ ^[0-9]+$ ]]; then
    echo "[STORAGE] ERROR: REMOTE_STORAGE_GUARD_INTERVAL_SECS must be an integer value (received '$REMOTE_STORAGE_GUARD_INTERVAL_SECS')." >&2
    exit 1
  fi
}

navdp_prepare_storage_runtime() {
  navdp_validate_storage_guard_settings
  REMOTE_SYNC_WORKER_PID=""
  REMOTE_SYNC_DONE_FILE=""
  REMOTE_STORAGE_UNAVAILABLE=false
  PARALLEL_PID=""
  PARALLEL_GROUP_ID=""
  STOP_REQUESTED=false
  REMOTE_STORAGE_GUARD_PID=""
  REMOTE_STORAGE_GUARD_STOP_FILE=""
}

handle_remote_storage_unavailable() {
  if [ "$REMOTE_STORAGE_UNAVAILABLE" = true ]; then
    return
  fi
  REMOTE_STORAGE_UNAVAILABLE=true
  echo "[STORAGE] Remote destination unavailable; pausing generation to avoid data loss." >&2
  if [ -n "$PARALLEL_GROUP_ID" ]; then
    kill -TERM -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
  elif [ -n "$PARALLEL_PID" ]; then
    kill -TERM "$PARALLEL_PID" >/dev/null 2>&1 || true
  fi
}

remote_sync_worker_loop() {
  local source_dir="$1"
  local remote_dir="$2"
  local done_flag="$3"
  local interval_secs="$4"
  local abort_on_failure="${5:-false}"
  local parent_pid="${6:-}"
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  echo "[STORAGE] Remote sync worker started for ${source_dir} -> ${REMOTE_SSH_TARGET:-?}:${remote_dir} (interval ${interval_secs}s)"
  local iteration=0
  while true; do
    iteration=$((iteration + 1))
    storage_sync_remote "$source_dir" "$remote_dir"
    local sync_status=$?
    if [ $sync_status -ne 0 ]; then
      echo "[STORAGE] WARN: Remote sync worker pass ${iteration} failed with status ${sync_status}." >&2
      if [ "$abort_on_failure" = true ]; then
        echo "[STORAGE] Remote sync worker detected unreachable destination; notifying renderer to pause." >&2
        if [ -n "$parent_pid" ]; then
          kill -s USR1 "$parent_pid" >/dev/null 2>&1 || true
        fi
        break
      fi
    fi
    if [ -f "$done_flag" ] && [ $sync_status -eq 0 ]; then
      echo "[STORAGE] Remote sync worker confirmed final sync after ${iteration} pass(es)."
      break
    fi
    sleep "$interval_secs"
  done
  echo "[STORAGE] Remote sync worker exiting."
}

start_remote_sync_worker() {
  local source_dir="$1"
  local remote_dir="$2"
  local interval_secs="$3"
  local abort_on_failure="${4:-false}"
  local parent_pid="${5:-}"
  REMOTE_SYNC_DONE_FILE=$(mktemp "${TMPDIR:-/tmp}/remote_sync_done.XXXXXX") || return 1
  rm -f "$REMOTE_SYNC_DONE_FILE"
  remote_sync_worker_loop "$source_dir" "$remote_dir" "$REMOTE_SYNC_DONE_FILE" "$interval_secs" "$abort_on_failure" "$parent_pid" &
  REMOTE_SYNC_WORKER_PID=$!
}

signal_remote_sync_completion() {
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    : > "$REMOTE_SYNC_DONE_FILE"
  fi
}

wait_remote_sync_worker() {
  if [ -n "$REMOTE_SYNC_WORKER_PID" ]; then
    wait "$REMOTE_SYNC_WORKER_PID" || true
    REMOTE_SYNC_WORKER_PID=""
  fi
  if [ -n "$REMOTE_SYNC_DONE_FILE" ]; then
    rm -f "$REMOTE_SYNC_DONE_FILE"
    REMOTE_SYNC_DONE_FILE=""
  fi
}

storage_guard_loop() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_bytes="$3"
  local resume_bytes="$4"
  local interval_secs="$5"
  local stop_flag="$6"
  local limit_gb="$7"
  local resume_gb="$8"
  local paused=false
  if [ -z "$interval_secs" ] || [ "$interval_secs" -le 0 ]; then
    interval_secs=60
  fi
  while true; do
    if [ -n "$stop_flag" ] && [ -f "$stop_flag" ]; then
      break
    fi
    if [ -z "$pgid" ] || ! kill -0 "$pgid" >/dev/null 2>&1; then
      break
    fi
    local usage_bytes
    usage_bytes=$(du -sb "$watch_dir" 2>/dev/null | awk '{print $1}')
    if [ -z "$usage_bytes" ]; then
      usage_bytes=0
    fi
    if [ "$usage_bytes" -ge "$limit_bytes" ]; then
      if [ "$paused" = false ]; then
        local usage_gb
        usage_gb=$(awk -v b="$usage_bytes" 'BEGIN { printf("%.2f", b / (1024*1024*1024)) }')
        echo "[STORAGE] Local usage ${usage_gb}GB reached limit ${limit_gb}GB; pausing generation until below ${resume_gb}GB."
        kill -STOP -- "-$pgid" >/dev/null 2>&1 || true
        paused=true
      fi
    elif [ "$paused" = true ] && [ "$usage_bytes" -le "$resume_bytes" ]; then
      echo "[STORAGE] Local usage dropped below ${resume_gb}GB; resuming generation."
      kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
      paused=false
    fi
    sleep "$interval_secs"
  done
  if [ "$paused" = true ] && [ -n "$pgid" ]; then
    kill -CONT -- "-$pgid" >/dev/null 2>&1 || true
  fi
}

start_storage_guard() {
  local watch_dir="$1"
  local pgid="$2"
  local limit_gb="$3"
  local resume_gb="$4"
  local interval_secs="$5"
  local limit_bytes=$((limit_gb * 1024 * 1024 * 1024))
  local resume_bytes=$((resume_gb * 1024 * 1024 * 1024))
  REMOTE_STORAGE_GUARD_STOP_FILE=$(mktemp "${TMPDIR:-/tmp}/storage_guard_stop.XXXXXX") || return 1
  rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
  storage_guard_loop "$watch_dir" "$pgid" "$limit_bytes" "$resume_bytes" "$interval_secs" "$REMOTE_STORAGE_GUARD_STOP_FILE" "$limit_gb" "$resume_gb" &
  REMOTE_STORAGE_GUARD_PID=$!
}

stop_storage_guard() {
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    : > "$REMOTE_STORAGE_GUARD_STOP_FILE"
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_PID" ]; then
    wait "$REMOTE_STORAGE_GUARD_PID" || true
    REMOTE_STORAGE_GUARD_PID=""
  fi
  if [ -n "$REMOTE_STORAGE_GUARD_STOP_FILE" ]; then
    rm -f "$REMOTE_STORAGE_GUARD_STOP_FILE"
    REMOTE_STORAGE_GUARD_STOP_FILE=""
  fi
}

cleanup_run() {
  wait_remote_sync_worker
  stop_storage_guard
}

handle_interrupt() {
  if [ "$STOP_REQUESTED" = true ]; then
    return
  fi
  STOP_REQUESTED=true
  echo "[RUN] Interrupt received; stopping workers..." >&2
  trap - EXIT
  if [ -n "$PARALLEL_GROUP_ID" ]; then
    kill -INT -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
    sleep 1
    kill -TERM -- "-$PARALLEL_GROUP_ID" >/dev/null 2>&1 || true
  elif [ -n "$PARALLEL_PID" ]; then
    kill -INT "$PARALLEL_PID" >/dev/null 2>&1 || true
    sleep 1
    kill -TERM "$PARALLEL_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$PARALLEL_PID" ]; then
    wait "$PARALLEL_PID" >/dev/null 2>&1 || true
    PARALLEL_PID=""
  fi
  if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    signal_remote_sync_completion
  fi
  cleanup_run
  exit 130
}

navdp_install_storage_traps() {
  trap cleanup_run EXIT
  trap handle_remote_storage_unavailable USR1
  trap handle_interrupt INT TERM
}

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

prepare_local_output_dir() {
  local target="$1"
  ensure_writable_dir "$target"
  if storage_bool_true "$CLEAR_LOCAL_OUTPUT_DIR"; then
    echo "[CLEAN] Clearing previous contents under ${target}"
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

navdp_check_storage_destinations() {
  if storage_bool_true "$ENABLE_NAS_STORAGE"; then
    if [ -z "$OFFLOAD_NAS_DIR" ]; then
      echo "[CHECK] ERROR: ENABLE_NAS_STORAGE=true requires OFFLOAD_NAS_DIR." >&2
      exit 1
    fi
    local nas_test_dir="${OFFLOAD_NAS_DIR}/__connectivity_check__"
    if mkdir -p "${nas_test_dir}" \
      && : > "${nas_test_dir}/.touch" \
      && rm -f "${nas_test_dir}/.touch"; then
      echo "[CHECK] NAS reachable at ${OFFLOAD_NAS_DIR}"
    else
      echo "[CHECK] ERROR: cannot write to NAS ${OFFLOAD_NAS_DIR}" >&2
      exit 1
    fi
  fi

  if storage_bool_true "$ENABLE_REMOTE_STORAGE"; then
    if ! storage_test_remote_connection "$REMOTE_TARGET_DIR"; then
      if [ "$REMOTE_ONLY_STORAGE" = true ]; then
        echo "[CHECK] ERROR: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable and no alternate storage is configured." >&2
        exit 1
      else
        echo "[CHECK] WARN: remote destination ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR} is unreachable; continuing with other storage backends." >&2
      fi
    else
      echo "[CHECK] Remote destination reachable at ${REMOTE_SSH_TARGET:-?}:${REMOTE_TARGET_DIR}"
    fi
  fi

  echo "[CONFIG] ENABLE_LOCAL_STORAGE=${ENABLE_LOCAL_STORAGE}"
  echo "[CONFIG] ENABLE_NAS_STORAGE=${ENABLE_NAS_STORAGE}"
  echo "[CONFIG] ENABLE_REMOTE_STORAGE=${ENABLE_REMOTE_STORAGE}"
}
