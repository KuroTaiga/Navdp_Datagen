#!/usr/bin/env bash
set -euo pipefail

# Run from the Navdp_Datagen repo root on 5880.
# Override with environment variables, e.g. WORKERS=4 FRAMES=16 CLEAN=1 bash ...

WORKERS="${WORKERS:-4}"
FRAMES="${FRAMES:-16}"
GPU_ID="${GPU_ID:-0}"
CLEAN="${CLEAN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.1}"
COMMAND_ATTEMPTS="${COMMAND_ATTEMPTS:-3}"
GROUP_MAX_LABELS_PER_COMMAND="${GROUP_MAX_LABELS_PER_COMMAND:-0}"
MAX_ITEMS_PER_CHUNK="${MAX_ITEMS_PER_CHUNK:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/dongjk/.conda/envs/cuda121/bin/python}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/mnt/DATA1/dongjk/navdp_data/h100_phase1_baseline_9891557/package_baseline_50}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/mnt/DATA1/dongjk/navdp_data}"
GIT_SHA="$(git rev-parse --short HEAD)"
ROOT="${ROOT:-${REMOTE_DATA_ROOT}/massgen_full50_persistent_multiworker_${GIT_SHA}_w${WORKERS}_f${FRAMES}}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/run_w${WORKERS}}"

mkdir -p "${ROOT}/logs"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "repo=$(pwd)"
  echo "git_sha=${GIT_SHA}"
  echo "workers=${WORKERS}"
  echo "gpu_id=${GPU_ID}"
  echo "frames=${FRAMES}"
  echo "package_root=${PACKAGE_ROOT}"
  echo "root=${ROOT}"
  echo "run_root=${RUN_ROOT}"
} | tee "${ROOT}/run_config.env"

df -h > "${ROOT}/df_before.txt" || true
nvidia-smi > "${ROOT}/nvidia_smi_before.txt" || true
git status --short --branch > "${ROOT}/git_status_before.txt" || true

PLAN_ARGS=(
  scripts/massgen/plan_persistent_h100_schedule.py
  --package-root "${PACKAGE_ROOT}"
  --materialized-root "${ROOT}/materialized"
  --render-plan-output-json "${ROOT}/aggregate_render_plan.json"
  --output-json "${ROOT}/persistent_schedule.json"
  --renders-per-family-source-scene 50
  --family deliver_to_human
  --family dense_dynamic_avoidance
  --family dense_dynamic_humans
  --family human_guided_uncertain_region
  --family navigate_with_social_constraints:pedestrian_yield
  --family navigate_with_social_constraints:queue_order
  --family serve_queue
  --python-bin "${PYTHON_BIN}"
  --render-script "$(pwd)/render_label_paths_telesim.py"
  --video-backend cpu
  --device cuda
  --minimal-frames "${FRAMES}"
  --actor-gpu-resident
  --actor-runtime-cache
  --gpu-id "${GPU_ID}"
  --workers-per-gpu "${WORKERS}"
  --max-items-per-chunk "${MAX_ITEMS_PER_CHUNK}"
  --include-execution
)

echo "[PLAN] ${PYTHON_BIN} ${PLAN_ARGS[*]}" | tee "${ROOT}/logs/plan.log"
"${PYTHON_BIN}" "${PLAN_ARGS[@]}" 2>&1 | tee -a "${ROOT}/logs/plan.log"

"${PYTHON_BIN}" scripts/massgen/summarize_persistent_schedule_lengths.py \
  --aggregate-render-plan-json "${ROOT}/aggregate_render_plan.json" \
  --schedule-json "${ROOT}/persistent_schedule.json" \
  --output-json "${ROOT}/natural_length_projection.json" \
  2>&1 | tee "${ROOT}/logs/natural_lengths.log"

if [[ "${RUN_RENDER}" == "1" ]]; then
  RUN_ARGS=(
    scripts/massgen/run_persistent_h100_schedule.py
    --schedule-json "${ROOT}/persistent_schedule.json"
    --results-root "${RUN_ROOT}"
    --repo-root "$(pwd)"
    --workers "${WORKERS}"
    --group-max-labels-per-command "${GROUP_MAX_LABELS_PER_COMMAND}"
    --command-attempts "${COMMAND_ATTEMPTS}"
    --gpu-sample-interval-sec "${SAMPLE_INTERVAL_SEC}"
    --preemptible-output
    --resume
  )
  if [[ "${CLEAN}" == "1" ]]; then
    RUN_ARGS+=(--clean)
  fi
  echo "[RUN] ${PYTHON_BIN} ${RUN_ARGS[*]}" | tee "${ROOT}/logs/run.log"
  "${PYTHON_BIN}" "${RUN_ARGS[@]}" 2>&1 | tee -a "${ROOT}/logs/run.log"
fi

if [[ "${RUN_ANALYSIS}" == "1" && -f "${RUN_ROOT}/benchmark_summary.json" ]]; then
  export MPLCONFIGDIR="${ROOT}/mplconfig"
  export XDG_CACHE_HOME="${ROOT}/xdg_cache"
  export FC_CACHEDIR="${ROOT}/fontconfig"
  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${FC_CACHEDIR}"
  "${PYTHON_BIN}" scripts/massgen/report_persistent_h100_schedule_run.py \
    --run-root "${RUN_ROOT}" \
    --output-root "${ROOT}/report_w${WORKERS}" \
    --title "MassGen Full50 Persistent Multiworker 5880 w${WORKERS}" \
    --stage-window-min 4 \
    --natural-length-json "${ROOT}/natural_length_projection.json" \
    2>&1 | tee "${ROOT}/logs/report.log"
fi

df -h > "${ROOT}/df_after.txt" || true
nvidia-smi > "${ROOT}/nvidia_smi_after.txt" || true
find "${RUN_ROOT}" -name '*.mp4' | sort > "${ROOT}/mp4_files.txt" || true
wc -l "${ROOT}/mp4_files.txt" | tee "${ROOT}/mp4_count.txt" || true

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "${ROOT}/run_config.env"
echo "root=${ROOT}"
