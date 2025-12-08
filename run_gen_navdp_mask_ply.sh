#!/bin/bash

INPUT_DIR="./data/scenes"
OUTPUT_DIR="/mnt/nas/zhangxiaotao/scenes_projection_fixed"
MAX_SCENES= ""

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting scene processing..."
if [ -n "$MAX_SCENES" ]; then
    python3 gen_navdp_mask_ply.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --max-scenes "$MAX_SCENES"
else
    python3 gen_navdp_mask_ply.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR"
fi

echo "Processing completed, results saved in $OUTPUT_DIR"
