#!/usr/bin/env bash
# Evaluate the character recognition results.
#
#   bash scripts/eval.sh            # visual + audio
#   bash scripts/eval.sh visual     # character box mIoU and character name AP
#   bash scripts/eval.sh audio      # audio recognition AP on GT time segments
#
# The audio evaluation uses the refined predictions when they exist, otherwise
# the raw ones; override with AUD_EVAL_FILE=/path/to/predictions.json.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

TARGET="${1:-all}"
cd "$CR_DIR"

if [ "$TARGET" = "all" ] || [ "$TARGET" = "visual" ]; then
    require_path "$VIS_SELECTED_FILE" "visual prediction file (VIS_SELECTED_FILE)"
    require_path "$ANNOTATION_FILE" "box annotation file (ANNOTATION_FILE)"

    echo "=== Character box mIoU ==="
    python eval/eval_box.py \
        --prediction_file "$VIS_SELECTED_FILE" \
        --annotation_file "$ANNOTATION_FILE"

    echo "=== Character name AP (MovieNet style) ==="
    python eval/eval_name.py \
        --num_shot "$EVAL_NUM_SHOT" \
        --prediction_file "$VIS_SELECTED_FILE" \
        --annotation_file "$ANNOTATION_FILE"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "audio" ]; then
    if [ -z "${AUD_EVAL_FILE:-}" ]; then
        if [ -f "$AUD_REFINED_FILE" ]; then
            AUD_EVAL_FILE="$AUD_REFINED_FILE"
        else
            AUD_EVAL_FILE="$AUD_PRED_FILE"
        fi
    fi
    require_path "$AUD_EVAL_FILE" "audio prediction file (AUD_EVAL_FILE)"

    OSR_ARGS=()
    if [ "$EVAL_OSR" = "1" ]; then
        OSR_ARGS=(--osr)
    fi

    echo "=== Audio recognition AP (${AUD_EVAL_FILE}) ==="
    python eval/eval_asr.py \
        --prediction_file "$AUD_EVAL_FILE" \
        "${OSR_ARGS[@]}"
fi

if [ "$TARGET" != "all" ] && [ "$TARGET" != "visual" ] && [ "$TARGET" != "audio" ]; then
    echo "usage: bash scripts/eval.sh [all|visual|audio]" >&2
    exit 1
fi
