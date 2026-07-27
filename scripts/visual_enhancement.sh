#!/usr/bin/env bash
# Refine the audio character recognition with the visual recognition results.
#
# visual_enhancement.py already supports chunking (--num_chunks / --chunk_idx),
# so the clips are split across the GPUs listed in $GPUS and the per-chunk
# results (written as <save_folder>/<movie_title>/<clip_idx>.jsonl) are merged
# by visual_enhancement_postprocessing.py.
#
#   bash scripts/visual_enhancement.sh
#   GPUS=0,1,2,3 bash scripts/visual_enhancement.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_path "$SYNC_CHECKPOINT" "synchronisation checkpoint (SYNC_CHECKPOINT)"
require_path "$AUD_PRED_FILE" "audio prediction file (AUD_PRED_FILE)"
require_path "$VIS_CLASSIFIED_FILE" "classified track file (VIS_CLASSIFIED_FILE)"
require_path "$VID_DIR" "video directory (VID_DIR)"
require_path "$FRAME_DIR" "extracted frame directory (FRAME_DIR)"

read_gpus
NUM_CHUNKS="${#GPU_LIST[@]}"

LOG_DIR="$WORK_DIR/logs"
mkdir -p "$SHOT_VID_DIR" "$SHOT_AUD_DIR" "$VE_OUTPUT_DIR" "$VE_RESULT_DIR" "$LOG_DIR" "$OUTPUT_DIR"
cd "$CR_DIR"

run_chunk() {
    local chunk="$1" gpu="$2"

    python audio_recognition/visual_enhancement.py \
        --gpu_id "$gpu" \
        --resume "$SYNC_CHECKPOINT" \
        --audio_prediction_file "$AUD_PRED_FILE" \
        --track_file "$VIS_CLASSIFIED_FILE" \
        --vid_dir "$VID_DIR" \
        --shot_vid_dir "$SHOT_VID_DIR" \
        --shot_aud_dir "$SHOT_AUD_DIR" \
        --frame_dir "$FRAME_DIR" \
        --output_dir "$VE_OUTPUT_DIR" \
        --save_folder "$VE_RESULT_DIR" \
        --num_chunks "$NUM_CHUNKS" \
        --chunk_idx "$chunk"
}

PIDS=()
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    gpu="${GPU_LIST[$chunk]}"
    log_file="$LOG_DIR/enhancement_chunk_${chunk}.log"
    echo "  chunk ${chunk}/${NUM_CHUNKS} -> gpu ${gpu}, log: ${log_file}"
    run_chunk "$chunk" "$gpu" > "$log_file" 2>&1 &
    PIDS+=("$!")
done

FAILED=()
for chunk in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$chunk]}"; then
        FAILED+=("$chunk")
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "error: chunk(s) ${FAILED[*]} failed, see ${LOG_DIR}/enhancement_chunk_<id>.log" >&2
    exit 1
fi

echo "Merging the visually enhanced predictions"
python audio_recognition/visual_enhancement_postprocessing.py \
    --result_dir "$VE_RESULT_DIR" \
    --save_file "$AUD_REFINED_FILE" \
    --threshold "$VE_THRESHOLD" \
    --alpha "$VE_ALPHA"

echo "Refined audio predictions: ${AUD_REFINED_FILE}"
echo "Next: PREDICTION_FILE=${AUD_REFINED_FILE} bash scripts/eval.sh"
