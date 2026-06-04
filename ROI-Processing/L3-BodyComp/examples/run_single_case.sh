#!/usr/bin/env bash
set -e

INPUT_NII="${1:-sample_data/sample_001.nii.gz}"
VOXTELL_MODEL="${2:-/path/to/voxtell_v1.1}"
OUTPUT_DIR="${3:-outputs/sample_001}"

mkdir -p "$OUTPUT_DIR"

python -m voxtell.applications.l3_body_composition \
  -i "$INPUT_NII" \
  -m "$VOXTELL_MODEL" \
  -o "$OUTPUT_DIR" \
  --device cuda
