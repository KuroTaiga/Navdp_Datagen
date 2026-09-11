#!/usr/bin/env bash
set -uo pipefail

CODE_ROOT="${CODE_ROOT:?set CODE_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
FAST_ROOT="${FAST_ROOT:?set FAST_ROOT}"
SLOW_ROOT="${SLOW_ROOT:?set SLOW_ROOT}"
SOCIAL_ROOT="${SOCIAL_ROOT:?set SOCIAL_ROOT}"
WORKERS="${WORKERS:-119}"
PATHS_PER_COHORT="${PATHS_PER_COHORT:-50}"
SCENES_PER_FAMILY="${SCENES_PER_FAMILY:-0}"
EXPERIMENT="${EXPERIMENT:-}"
export OUTPUT_DIR

mkdir -p "$OUTPUT_DIR"
python3 -c 'import json,os,datetime; p={"status":"running","started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getppid()}; open(os.environ["OUTPUT_DIR"]+"/status.json","w").write(json.dumps(p,indent=2)+"\n")'

cd "$CODE_ROOT"
experiment_args=()
if [[ -n "$EXPERIMENT" ]]; then
  experiment_args=(--experiment "$EXPERIMENT")
fi
python3 scripts/analysis/survey_poi_policy_matrix.py \
  --formal-fast-root "$FAST_ROOT" \
  --formal-slow-root "$SLOW_ROOT" \
  --formal-social-root "$SOCIAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --paths-per-cohort "$PATHS_PER_COHORT" \
  --scenes-per-family "$SCENES_PER_FAMILY" \
  --seed 20260911 \
  --workers "$WORKERS" \
  "${experiment_args[@]}"
exit_code=$?

if [[ "$exit_code" -eq 0 ]]; then
  final_status="success"
else
  final_status="failed"
fi
FINAL_STATUS="$final_status" EXIT_CODE="$exit_code" python3 -c 'import json,os,datetime; p={"status":os.environ["FINAL_STATUS"],"finished_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"exit_code":int(os.environ["EXIT_CODE"])}; open(os.environ["OUTPUT_DIR"]+"/status.json","w").write(json.dumps(p,indent=2)+"\n")'
exit "$exit_code"
