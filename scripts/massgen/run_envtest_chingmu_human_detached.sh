#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/team/telenav/code/Navdp_Datagen}"
PYTHON_BIN="${PYTHON_BIN:-/team/telenav/code/conda_envs/navdp_cuda121/bin/python}"
MISSION_CONFIG="${MISSION_CONFIG:-${REPO_ROOT}/configs/massgen/envtest_chingmu_human_missions.env}"

SOURCE_PACKAGE="${SOURCE_PACKAGE:-/team/telenav/massgen_packages/h100_phase1_baseline_9891557/package_formal_chingmu15_100_h100_good_actions_seed20260825}"
OUTPUT_BASE="${OUTPUT_BASE:-/private/dongjk/navdata}"
PACKAGE_BASE="${PACKAGE_BASE:-${OUTPUT_BASE}/massgen_packages}"
RUN_BASE="${RUN_BASE:-${OUTPUT_BASE}/massgen_runs}"
ACTOR_ROOT="${ACTOR_ROOT:-/team/telenav/human_avatars/20260811_stmc_kimodo_new_actions/grouped_actions/use_default_no_waving}"
ASSET_SELECTION_SEED="${ASSET_SELECTION_SEED:-20260826}"

GPU_DEVICES="${GPU_DEVICES:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-16}"
CPU_CORES="${CPU_CORES:-$(nproc 2>/dev/null || printf '192')}"
VIDEO_BACKEND="${VIDEO_BACKEND:-cpu}"
MAX_ITEMS_PER_CHUNK="${MAX_ITEMS_PER_CHUNK:-25}"
GROUP_MAX_LABELS_PER_COMMAND="${GROUP_MAX_LABELS_PER_COMMAND:-25}"
COMMAND_ATTEMPTS="${COMMAND_ATTEMPTS:-3}"
GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-1.0}"
MONITOR_POLL_SEC="${MONITOR_POLL_SEC:-60}"
COMPACT_POLL_SEC="${COMPACT_POLL_SEC:-300}"
GPU_SAMPLE_STRIDE="${GPU_SAMPLE_STRIDE:-30}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"
LOG_TAIL_BYTES="${LOG_TAIL_BYTES:-262144}"
STAGE_WINDOW_MIN="${STAGE_WINDOW_MIN:-10}"
SCENE_AFFINITY="${SCENE_AFFINITY:-0}"
RENDERS_PER_FAMILY_SOURCE_SCENE="${RENDERS_PER_FAMILY_SOURCE_SCENE:-100}"
MAX_RENDERS="${MAX_RENDERS:-0}"
MINIMAL_FRAMES="${MINIMAL_FRAMES:-0}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/team/telenav/h100_results/baselines/5880_5548b12_w4_f16_benchmark_summary.json}"

cd "${REPO_ROOT}"
if [[ ! -f "${MISSION_CONFIG}" ]]; then
  echo "missing mission config: ${MISSION_CONFIG}" >&2
  exit 66
fi
source "${MISSION_CONFIG}"

if (( ${#ACTIVE_MISSION_FAMILIES[@]} == 0 )); then
  echo "ACTIVE_MISSION_FAMILIES is empty in ${MISSION_CONFIG}" >&2
  exit 64
fi
if (( ${#ACTIVE_SOURCES[@]} == 0 )); then
  echo "ACTIVE_SOURCES is empty in ${MISSION_CONFIG}" >&2
  exit 64
fi

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
RUN_NAME="${RUN_NAME:-envtest_chingmu_human_random_seed${ASSET_SELECTION_SEED}_${GIT_SHA}_w${JOBS_PER_GPU}x$(tr ',' '_' <<< "${GPU_DEVICES}")_c${GROUP_MAX_LABELS_PER_COMMAND}_f${MINIMAL_FRAMES}}"
RANDOMIZED_PACKAGE="${RANDOMIZED_PACKAGE:-${PACKAGE_BASE}/package_formal_chingmu15_100_envtest_human_random_seed${ASSET_SELECTION_SEED}}"
RESULT_ROOT="${RESULT_ROOT:-${RUN_BASE}/${RUN_NAME}}"
LAUNCH_LOG="${RESULT_ROOT}/launcher.log"

mkdir -p "${PACKAGE_BASE}" "${RUN_BASE}" "${RESULT_ROOT}/workflow" "${RESULT_ROOT}/monitor"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${LAUNCH_LOG}"
}

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

family_args=()
for family in "${ACTIVE_MISSION_FAMILIES[@]}"; do
  family_args+=(--family "${family}")
done

source_args=()
for source in "${ACTIVE_SOURCES[@]}"; do
  source_args+=(--source "${source}")
done

scene_args=()
for scene in "${ACTIVE_SCENES[@]}"; do
  scene_args+=(--scene "${scene}")
done

log "envtest launcher started"
log "repo=${REPO_ROOT}"
log "git_sha=${GIT_SHA}"
log "mission_config=${MISSION_CONFIG}"
log "active_families=${ACTIVE_MISSION_FAMILIES[*]}"
log "active_sources=${ACTIVE_SOURCES[*]}"
log "active_scenes=${ACTIVE_SCENES[*]:-all selected scenes}"
log "source_package=${SOURCE_PACKAGE}"
log "randomized_package=${RANDOMIZED_PACKAGE}"
log "result_root=${RESULT_ROOT}"
log "actor_root=${ACTOR_ROOT}"
log "gpu_devices=${GPU_DEVICES} jobs_per_gpu=${JOBS_PER_GPU} cpu_cores=${CPU_CORES}"

df -h "${OUTPUT_BASE}" > "${RESULT_ROOT}/workflow/df_before.txt" 2>&1 || true
nvidia-smi > "${RESULT_ROOT}/workflow/nvidia_smi_before.txt" 2>&1 || true
git status --short --branch > "${RESULT_ROOT}/workflow/git_status_before.txt" 2>&1 || true

if [[ ! -f "${RANDOMIZED_PACKAGE}/smoketest_package_index.json" ]]; then
  if [[ -e "${RANDOMIZED_PACKAGE}" ]]; then
    log "refusing to reuse incomplete randomized package: ${RANDOMIZED_PACKAGE}"
    exit 65
  fi
  log "building randomized active package"
  "${PYTHON_BIN}" scripts/massgen/randomize_h100_package_human_assets.py \
    --source-package "${SOURCE_PACKAGE}" \
    --output-package "${RANDOMIZED_PACKAGE}" \
    --actor-root "${ACTOR_ROOT}" \
    --seed "${ASSET_SELECTION_SEED}" \
    --max-entries-per-family-source-scene "${RENDERS_PER_FAMILY_SOURCE_SCENE}" \
    "${family_args[@]}" \
    "${source_args[@]}" \
    "${scene_args[@]}" \
    > "${RESULT_ROOT}/workflow/randomize_package.log" 2>&1
else
  log "reusing randomized active package"
fi

if pid_alive "${RESULT_ROOT}/driver.pid"; then
  log "driver already running pid=$(cat "${RESULT_ROOT}/driver.pid")"
else
  run_args=(
    "${PYTHON_BIN}"
    scripts/massgen/run_family_rollout_h100.py
    --package-root "${RANDOMIZED_PACKAGE}"
    --results-root "${RESULT_ROOT}"
    --python-bin "${PYTHON_BIN}"
    --gpu-devices "${GPU_DEVICES}"
    --jobs-per-gpu "${JOBS_PER_GPU}"
    --cpu-cores "${CPU_CORES}"
    --video-backend "${VIDEO_BACKEND}"
    --renders-per-family-source-scene "${RENDERS_PER_FAMILY_SOURCE_SCENE}"
    --max-renders "${MAX_RENDERS}"
    --max-items-per-chunk "${MAX_ITEMS_PER_CHUNK}"
    --group-max-labels-per-command "${GROUP_MAX_LABELS_PER_COMMAND}"
    --command-attempts "${COMMAND_ATTEMPTS}"
    --util-sample-interval-sec "${GPU_SAMPLE_INTERVAL_SEC}"
    --stage-window-min "${STAGE_WINDOW_MIN}"
  )
  if (( MINIMAL_FRAMES > 0 )); then
    run_args+=(--minimal-frames "${MINIMAL_FRAMES}")
  fi
  if [[ -f "${BASELINE_SUMMARY}" ]]; then
    run_args+=(--baseline-summary "${BASELINE_SUMMARY}")
  fi
  if [[ "${SCENE_AFFINITY}" == "1" ]]; then
    run_args+=(--scene-affinity)
  else
    run_args+=(--no-scene-affinity)
  fi

  printf '%q ' "${run_args[@]}" > "${RESULT_ROOT}/workflow/driver_command.sh"
  printf '\n' >> "${RESULT_ROOT}/workflow/driver_command.sh"
  chmod +x "${RESULT_ROOT}/workflow/driver_command.sh"
  log "launching detached driver"
  nohup "${run_args[@]}" > "${RESULT_ROOT}/driver.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RESULT_ROOT}/driver.pid"
  log "driver pid=$(cat "${RESULT_ROOT}/driver.pid")"
fi

if pid_alive "${RESULT_ROOT}/monitor/monitor.pid"; then
  log "monitor already running pid=$(cat "${RESULT_ROOT}/monitor/monitor.pid")"
else
  log "launching detached monitor"
  nohup "${PYTHON_BIN}" scripts/massgen/monitor_h100_run.py \
    --results-root "${RESULT_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --poll-sec "${MONITOR_POLL_SEC}" \
    --stage-window-min "${STAGE_WINDOW_MIN}" \
    --title "Envtest CHINGMU Human MassGen" \
    > "${RESULT_ROOT}/monitor/monitor.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RESULT_ROOT}/monitor/monitor.pid"
  log "monitor pid=$(cat "${RESULT_ROOT}/monitor/monitor.pid")"
fi

if pid_alive "${RESULT_ROOT}/monitor/compactor.pid"; then
  log "compactor already running pid=$(cat "${RESULT_ROOT}/monitor/compactor.pid")"
else
  log "launching detached artifact compactor"
  nohup env \
    REPO_ROOT="${REPO_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    COMPACT_POLL_SEC="${COMPACT_POLL_SEC}" \
    MAX_LOG_BYTES="${MAX_LOG_BYTES}" \
    LOG_TAIL_BYTES="${LOG_TAIL_BYTES}" \
    GPU_SAMPLE_STRIDE="${GPU_SAMPLE_STRIDE}" \
    bash -c '
      set -euo pipefail
      cd "${REPO_ROOT}"
      while true; do
        "${PYTHON_BIN}" scripts/massgen/compact_h100_run_artifacts.py \
          --results-root "${RESULT_ROOT}" \
          --max-log-bytes "${MAX_LOG_BYTES}" \
          --tail-bytes "${LOG_TAIL_BYTES}" \
          --gpu-sample-stride "${GPU_SAMPLE_STRIDE}" || true
        if [[ -f "${RESULT_ROOT}/driver.pid" ]]; then
          driver_pid="$(cat "${RESULT_ROOT}/driver.pid")"
          if [[ "${driver_pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${driver_pid}" >/dev/null 2>&1; then
            break
          fi
        fi
        sleep "${COMPACT_POLL_SEC}"
      done
      "${PYTHON_BIN}" scripts/massgen/compact_h100_run_artifacts.py \
        --results-root "${RESULT_ROOT}" \
        --max-log-bytes "${MAX_LOG_BYTES}" \
        --tail-bytes "${LOG_TAIL_BYTES}" \
        --gpu-sample-stride "${GPU_SAMPLE_STRIDE}" || true
    ' > "${RESULT_ROOT}/monitor/compactor.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RESULT_ROOT}/monitor/compactor.pid"
  log "compactor pid=$(cat "${RESULT_ROOT}/monitor/compactor.pid")"
fi

log "launcher finished"
log "progress_md=${RESULT_ROOT}/monitor/PROGRESS.md"
