#!/usr/bin/env bash
# Turn classified tracks into per-frame predictions, and select the frames that
# carry ground-truth boxes for the MovieNet-style evaluation on CMD-AM.
#
#   bash scripts/visual_postprocessing.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_path "$VIS_CLASSIFIED_FILE" "classified track file (VIS_CLASSIFIED_FILE)"
# postprocessing.py walks <source_dir>/<clip_idx>/<shot_idx>/*.jpg, i.e. it reads
# the frames written by TGRP rather than the raw videos.
require_path "$FRAME_DIR" "extracted frame directory (FRAME_DIR)"

mkdir -p "$OUTPUT_DIR"
cd "$CR_DIR"

echo "Converting tracks to frame-level predictions"
python visual_recognition/postprocessing.py \
    --source_dir "$FRAME_DIR" \
    --track_file "$VIS_CLASSIFIED_FILE" \
    --save_file "$VIS_FRAME_PRED_FILE"

echo "Selecting the annotated key frames of the CMD-AM subset"
require_path "$ANNOTATION_FILE" "box annotation file (ANNOTATION_FILE)"
python visual_recognition/select.py \
    --annotation_file "$ANNOTATION_FILE" \
    --track_predictions_file "$SELECT_INPUT_FILE" \
    --save_path "$VIS_SELECTED_FILE"

echo "Frame predictions:    ${VIS_FRAME_PRED_FILE}"
echo "Evaluation input:     ${VIS_SELECTED_FILE}"
echo "Next: bash scripts/eval.sh"
