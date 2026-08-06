#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python3}
INPUT_DIR=${INPUT_DIR:-"${REPO_ROOT}/data/scenes"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/out/scenes_projection_debug"}
MAX_SCENES=${MAX_SCENES:-}

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

args=(
    "${SCRIPT_DIR}/gen_navdp_mask_ply.py"
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
)
if [ -n "$MAX_SCENES" ]; then
    args+=(--max-scenes "$MAX_SCENES")
fi

echo "Starting scene processing..."
"$PYTHON_BIN" "${args[@]}"
echo "Processing completed, results saved in $OUTPUT_DIR"
