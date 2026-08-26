#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/team/telenav/code/Navdp_Datagen}"
PYTHON_BIN="${PYTHON_BIN:-/team/telenav/code/conda_envs/navdp_cuda121/bin/python}"
PACKAGE_BASE="${PACKAGE_BASE:-/team/telenav/massgen_packages/h100_phase1_baseline_9891557}"
RAW_PACKAGE="${RAW_PACKAGE:-${PACKAGE_BASE}/package_formal_chingmu15_100_raw_seed20260825}"
H100_PACKAGE="${H100_PACKAGE:-${PACKAGE_BASE}/package_formal_chingmu15_100_h100_good_actions_seed20260825}"
RESULT_ROOT="${RESULT_ROOT:-/team/telenav/h100_results/massgen_uncapped_h100_good_actions_seed20260825_chingmu15_100_w32_c25_g25_noaff_f16}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/team/telenav/h100_results/baselines/5880_5548b12_w4_f16_benchmark_summary.json}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-43200}"
POLL_SEC="${POLL_SEC:-60}"

cd "${REPO_ROOT}"

mkdir -p "${RESULT_ROOT}/workflow" "${RESULT_ROOT}/monitor"
WORKFLOW_LOG="${RESULT_ROOT}/workflow/workflow.log"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${WORKFLOW_LOG}"
}

wait_for_raw_package() {
  local start now elapsed
  start="$(date +%s)"
  while [[ ! -f "${RAW_PACKAGE}/smoketest_package_index.json" ]]; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > WAIT_TIMEOUT_SEC )); then
      log "timeout waiting for raw package index: ${RAW_PACKAGE}/smoketest_package_index.json"
      return 1
    fi
    log "waiting for raw package index, elapsed=${elapsed}s"
    sleep "${POLL_SEC}"
  done
  log "raw package ready: ${RAW_PACKAGE}"
}

build_h100_package() {
  if [[ -f "${H100_PACKAGE}/smoketest_package_index.json" ]]; then
    log "H100 package already exists, reusing: ${H100_PACKAGE}"
    return 0
  fi
  if [[ -e "${H100_PACKAGE}" ]]; then
    log "H100 package path exists without complete index; refusing to overwrite: ${H100_PACKAGE}"
    return 1
  fi
  log "building H100 package: ${H100_PACKAGE}"
  "${PYTHON_BIN}" scripts/massgen/create_h100_selected_package.py \
    --source-package "${RAW_PACKAGE}" \
    --output-package "${H100_PACKAGE}" \
    --scene-selection all \
    --max-scenes 15 \
    --require-readable-scene-ply \
    --expected-entry-count 10500 \
    --interiorgs-root /team/telenav/navsources/InteriorGS \
    > "${RESULT_ROOT}/workflow/create_h100_package.log" 2>&1
  log "H100 package ready: ${H100_PACKAGE}"
}

launch_driver() {
  if [[ -f "${RESULT_ROOT}/driver.pid" ]] && ps -p "$(cat "${RESULT_ROOT}/driver.pid")" >/dev/null 2>&1; then
    log "driver already running pid=$(cat "${RESULT_ROOT}/driver.pid")"
    return 0
  fi
  log "launching H100 render driver"
  nohup "${PYTHON_BIN}" scripts/massgen/run_family_rollout_h100.py \
    --package-root "${H100_PACKAGE}" \
    --results-root "${RESULT_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --gpu-devices 0,1 \
    --jobs-per-gpu 16 \
    --cpu-cores 192 \
    --video-backend cpu \
    --max-items-per-chunk 25 \
    --group-max-labels-per-command 25 \
    --command-attempts 3 \
    --util-sample-interval-sec 0.1 \
    --stage-window-min 10 \
    --baseline-summary "${BASELINE_SUMMARY}" \
    --no-scene-affinity \
    > "${RESULT_ROOT}/driver.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RESULT_ROOT}/driver.pid"
  log "driver pid=$(cat "${RESULT_ROOT}/driver.pid")"
}

launch_monitor() {
  if [[ -f "${RESULT_ROOT}/monitor/monitor.pid" ]] && ps -p "$(cat "${RESULT_ROOT}/monitor/monitor.pid")" >/dev/null 2>&1; then
    log "monitor already running pid=$(cat "${RESULT_ROOT}/monitor/monitor.pid")"
    return 0
  fi
  log "launching detached progress monitor"
  nohup "${PYTHON_BIN}" scripts/massgen/monitor_h100_run.py \
    --results-root "${RESULT_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --poll-sec "${POLL_SEC}" \
    --stage-window-min 10 \
    --baseline-summary "${BASELINE_SUMMARY}" \
    > "${RESULT_ROOT}/monitor/monitor.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RESULT_ROOT}/monitor/monitor.pid"
  log "monitor pid=$(cat "${RESULT_ROOT}/monitor/monitor.pid")"
}

log "workflow started"
wait_for_raw_package
build_h100_package
launch_driver
launch_monitor
log "workflow launched render and monitor; exiting launcher"
