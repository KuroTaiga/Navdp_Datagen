#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storage_targets.sh"

REMOTE_STORAGE_ROOT=${REMOTE_STORAGE_ROOT:-${REMOTE_OUTPUT_DIR:-/srv/navdp}}
TEST_FOLDER_NAME=${1:-_storage_connection_test}
REMOTE_TARGET_DIR="${REMOTE_STORAGE_ROOT%/}/${TEST_FOLDER_NAME}"
SPEED_TEST_MB=${SPEED_TEST_MB:-200}
MEASURE_SPEED=${MEASURE_SPEED:-true}

echo "[STORAGE] Using remote root ${REMOTE_STORAGE_ROOT%/} (override via REMOTE_STORAGE_ROOT)."
storage_test_remote_connection "$REMOTE_TARGET_DIR"

if storage_bool_true "$MEASURE_SPEED"; then
  echo "[STORAGE] Running upload/download throughput tests (${SPEED_TEST_MB} MB)..."
  storage_measure_transfer_speed upload "$REMOTE_TARGET_DIR" "$SPEED_TEST_MB"
  storage_measure_transfer_speed download "$REMOTE_TARGET_DIR" "$SPEED_TEST_MB"
else
  echo "[STORAGE] Skipping throughput test (MEASURE_SPEED=${MEASURE_SPEED})."
fi
