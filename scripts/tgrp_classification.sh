#!/usr/bin/env bash
# Track-Guided Region Proposal (TGRP) + character identification on several GPUs.
#
# The clips in $SOURCE_DIR are split round-robin into one shard per GPU. Each
# worker runs tgrp.py and then classification.py on its own shard, and the
# per-shard outputs are merged at the end. Shards are independent, so a failed
# worker can be resumed by re-running with GPUS set to the ids you want.
#
#   bash scripts/tgrp_classification.sh
#   GPUS=0,1,2,3 bash scripts/tgrp_classification.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_path "$SOURCE_DIR" "source video directory (SOURCE_DIR)"
require_path "$SAM2_CHECKPOINT" "SAM2 checkpoint (SAM2_CHECKPOINT)"
require_path "$DINOV2_CKPT_DIR" "finetuned DINOv2 weights (DINOV2_CKPT_DIR)"
require_path "$CHAR_FEAT_DIR" "appearance bank features (CHAR_FEAT_DIR)"

read_gpus
NUM_SHARDS="${#GPU_LIST[@]}"

SPLIT_DIR="$WORK_DIR/shards"
SHARD_OUT_DIR="$WORK_DIR/shard_outputs"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$SPLIT_DIR" "$SHARD_OUT_DIR" "$LOG_DIR" "$FRAME_DIR" "$OUTPUT_DIR"

MASK_ARGS=()
if [ "$USE_MASK" = "1" ]; then
    MASK_ARGS=(--mask)
fi

# ------------------------------------------------------- split the video files
# The shard directories only ever hold symlinks created here, so they are safe
# to clear between runs; the videos themselves are never touched.
find "$SPLIT_DIR" -mindepth 1 -maxdepth 2 -type l -delete

mapfile -t VIDEOS < <(find "$SOURCE_DIR" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.webm' \) | sort)

if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "error: no video files found in ${SOURCE_DIR}" >&2
    exit 1
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    mkdir -p "$SPLIT_DIR/shard_${shard}"
done

for idx in "${!VIDEOS[@]}"; do
    video="${VIDEOS[$idx]}"
    shard=$((idx % NUM_SHARDS))
    ln -sfn "$(readlink -f "$video")" "$SPLIT_DIR/shard_${shard}/$(basename "$video")"
done

echo "Split ${#VIDEOS[@]} clips into ${NUM_SHARDS} shards (GPUs: ${GPUS})"

# ------------------------------------------------------------ run the workers
cd "$CR_DIR"

run_shard() {
    local shard="$1" gpu="$2"
    local shard_src="$SPLIT_DIR/shard_${shard}"
    local track_file="$SHARD_OUT_DIR/tracks_shard_${shard}.jsonl"
    local classified_file="$SHARD_OUT_DIR/classified_shard_${shard}.json"

    echo "[shard ${shard} | gpu ${gpu}] track-guided region proposal"
    CUDA_VISIBLE_DEVICES="$gpu" python visual_recognition/tgrp.py \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --source_dir "$shard_src" \
        --save_file "$track_file"

    echo "[shard ${shard} | gpu ${gpu}] character identification"
    CUDA_VISIBLE_DEVICES="$gpu" python visual_recognition/classification.py \
        --model_size "$DINOV2_MODEL_SIZE" \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --ckpt_dir "$DINOV2_CKPT_DIR" \
        --char_feat_dir "$CHAR_FEAT_DIR" \
        --track_file "$track_file" \
        --frame_dir "$FRAME_DIR" \
        --save_file "$classified_file" \
        "${MASK_ARGS[@]}"
}

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPU_LIST[$shard]}"
    log_file="$LOG_DIR/shard_${shard}.log"
    echo "  shard ${shard} -> gpu ${gpu}, log: ${log_file}"
    run_shard "$shard" "$gpu" > "$log_file" 2>&1 &
    PIDS+=("$!")
done

FAILED=()
for shard in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$shard]}"; then
        FAILED+=("$shard")
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "error: shard(s) ${FAILED[*]} failed, see ${LOG_DIR}/shard_<id>.log" >&2
    exit 1
fi

# --------------------------------------------------------- merge the outputs
cat "$SHARD_OUT_DIR"/tracks_shard_*.jsonl > "$VIS_TRACK_FILE"

python - "$VIS_CLASSIFIED_FILE" "$SHARD_OUT_DIR"/classified_shard_*.json <<'PY'
import json
import sys

save_file, shard_files = sys.argv[1], sys.argv[2:]

merged = []
for shard_file in shard_files:
    with open(shard_file, "r") as infile:
        merged.extend(json.load(infile))

with open(save_file, "w") as outfile:
    json.dump(merged, outfile, indent=4)

print(f"Merged {len(shard_files)} shards ({len(merged)} shots) into {save_file}")
PY

echo "Region proposals:     ${VIS_TRACK_FILE}"
echo "Classified tracks:    ${VIS_CLASSIFIED_FILE}"
echo "Next: bash scripts/visual_postprocessing.sh"
