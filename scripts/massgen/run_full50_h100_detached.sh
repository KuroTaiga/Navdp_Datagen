#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/team/telenav/code/Navdp_Datagen}"
PYTHON_BIN="${PYTHON_BIN:-/team/telenav/code/conda_envs/navdp_cuda121/bin/python}"
PACKAGE_ROOT="${PACKAGE_ROOT:-/team/telenav/massgen_packages/h100_phase1_baseline_9891557/package_baseline_50_h100_good_actions_seed20260824_20scene_local_interiorgs}"
RESULT_ROOT="${RESULT_ROOT:-/team/telenav/h100_results/massgen_full50_h100_good_actions_seed20260824_20scene_local_interiorgs_d760287_w32_c25_g25_noaff_f16}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/team/telenav/h100_results/baselines/5880_5548b12_w4_f16_benchmark_summary.json}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-16}"
CPU_CORES="${CPU_CORES:-192}"
MAX_ITEMS_PER_CHUNK="${MAX_ITEMS_PER_CHUNK:-25}"
SCENE_AFFINITY="${SCENE_AFFINITY:-0}"
GROUP_MAX_LABELS_PER_COMMAND="${GROUP_MAX_LABELS_PER_COMMAND:-25}"
COMMAND_ATTEMPTS="${COMMAND_ATTEMPTS:-3}"
MINIMAL_FRAMES="${MINIMAL_FRAMES:-16}"
STAGE_WINDOW_MIN="${STAGE_WINDOW_MIN:-4}"
CLEAN="${CLEAN:-0}"

if [[ "${CLEAN}" == "1" ]]; then
  case "${RESULT_ROOT}" in
    /team/telenav/h100_results/*)
      rm -rf "${RESULT_ROOT}"
      ;;
    *)
      echo "Refusing to clean unexpected RESULT_ROOT=${RESULT_ROOT}" >&2
      exit 64
      ;;
  esac
fi
mkdir -p "${RESULT_ROOT}"
exec >> "${RESULT_ROOT}/driver.log" 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "repo=${REPO_ROOT}"
echo "package_root=${PACKAGE_ROOT}"
echo "result_root=${RESULT_ROOT}"
echo "gpu_devices=${GPU_DEVICES}"
echo "jobs_per_gpu=${JOBS_PER_GPU}"
echo "max_items_per_chunk=${MAX_ITEMS_PER_CHUNK}"
echo "scene_affinity=${SCENE_AFFINITY}"
echo "group_max_labels_per_command=${GROUP_MAX_LABELS_PER_COMMAND}"
echo "clean=${CLEAN}"

cd "${REPO_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  scripts/massgen/run_family_rollout_h100.py
  --package-root "${PACKAGE_ROOT}"
  --results-root "${RESULT_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --gpu-devices "${GPU_DEVICES}"
  --jobs-per-gpu "${JOBS_PER_GPU}"
  --cpu-cores "${CPU_CORES}"
  --video-backend cpu
  --renders-per-family-source-scene 50
  --minimal-frames "${MINIMAL_FRAMES}"
  --max-items-per-chunk "${MAX_ITEMS_PER_CHUNK}"
  --group-max-labels-per-command "${GROUP_MAX_LABELS_PER_COMMAND}"
  --command-attempts "${COMMAND_ATTEMPTS}"
  --util-sample-interval-sec 0.1
  --stage-window-min "${STAGE_WINDOW_MIN}"
  --baseline-summary "${BASELINE_SUMMARY}"
)

if [[ "${SCENE_AFFINITY}" == "1" ]]; then
  cmd+=(--scene-affinity)
else
  cmd+=(--no-scene-affinity)
fi

echo "command=${cmd[*]}"
"${cmd[@]}"
rc=$?
echo "finished_at=$(date --iso-8601=seconds)"
echo "returncode=${rc}"
exit "${rc}"
