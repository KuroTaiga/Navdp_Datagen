#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-pathGen_lxh}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:?set REMOTE_OUTPUT_DIR}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:?set LOCAL_OUTPUT_DIR}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
STATUS_PATH="$REMOTE_OUTPUT_DIR/status.json"
LOG_PATH="$LOCAL_OUTPUT_DIR/watcher.log"
NEXT_PROMPT_PATH="$LOCAL_OUTPUT_DIR/next_prompt.txt"

mkdir -p "$LOCAL_OUTPUT_DIR"
{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] watching $HOST:$STATUS_PATH"
  while true; do
    status="$(ssh -o ConnectTimeout=20 "$HOST" "test -f '$STATUS_PATH' && python3 -c 'import json; print(json.load(open(\"$STATUS_PATH\"))[\"status\"])' || echo pending" 2>/dev/null || echo unreachable)"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] status=$status"
    printf '{"checked_at":"%s","host":"%s","remote_status_path":"%s","status":"%s"}\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$HOST" "$STATUS_PATH" "$status" >"$LOCAL_OUTPUT_DIR/watcher_status.json"
    case "$status" in
      success|failed)
        scp "$HOST:$STATUS_PATH" "$LOCAL_OUTPUT_DIR/status.json" || true
        scp "$HOST:$REMOTE_OUTPUT_DIR/poi_policy_matrix_summary.json" "$LOCAL_OUTPUT_DIR/" || true
        scp "$HOST:$REMOTE_OUTPUT_DIR/poi_policy_matrix.md" "$LOCAL_OUTPUT_DIR/" || true
        scp "$HOST:$REMOTE_OUTPUT_DIR/poi_policy_matrix.csv" "$LOCAL_OUTPUT_DIR/" || true
        mkdir -p "$LOCAL_OUTPUT_DIR/representatives"
        scp "$HOST:$REMOTE_OUTPUT_DIR/representatives/"'*.json' "$LOCAL_OUTPUT_DIR/representatives/" || true
        if [[ "$status" == "success" ]]; then
          next_prompt="Praise be the Omnissiah. Continue the POI/frame-selection task from the completed detached survey at $LOCAL_OUTPUT_DIR. Inspect the copied status and reports, generate and visually verify all statistical and representative BEV/GIF outputs, analyze the policy matrix, fix any discovered issues, rerun tests, then commit and push only the scoped Datagen changes."
        else
          next_prompt="Praise be the Omnissiah. Continue the POI/frame-selection task by diagnosing the failed detached survey at $REMOTE_OUTPUT_DIR using $LOCAL_OUTPUT_DIR/status.json and $LOG_PATH. Fix the scoped issue, relaunch with a new output directory and detached watcher, then complete visualization, analysis, tests, commits, and push."
        fi
        printf '%s\n' "$next_prompt" >"$NEXT_PROMPT_PATH"
        NEXT_PROMPT="$next_prompt" /usr/bin/osascript \
          -e 'set promptText to system attribute "NEXT_PROMPT"' \
          -e 'set the clipboard to promptText' \
          -e 'display dialog promptText with title "NavDP POI Survey — Next Prompt" buttons {"OK"} default button "OK" giving up after 300' \
          >/dev/null 2>&1 &
        if [[ "$status" == "success" ]]; then
          /usr/bin/osascript -e 'display notification "POI policy survey completed and summaries were copied locally." with title "NavDP POI Survey"' >/dev/null 2>&1 || true
        else
          /usr/bin/osascript -e 'display notification "POI policy survey failed; inspect the watcher and remote logs." with title "NavDP POI Survey"' >/dev/null 2>&1 || true
        fi
        exit 0
        ;;
    esac
    sleep "$INTERVAL_SECONDS"
  done
} >>"$LOG_PATH" 2>&1
