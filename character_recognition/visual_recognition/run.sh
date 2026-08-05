#!/bin/bash
#
# Track-Guided Region Proposal over the CMDAM clips, sharded across a SLURM
# array (one GPU per task), then merged into a single JSONL.
#
# CMDAM stores its clips as vid/<year>/<clip_id>.mkv, but tgrp.py lists exactly
# one level of --source_dir, so each shard is handed a flat directory of
# symlinks to its share of the clips. Clip ids are unique across the year
# folders, so flattening loses nothing; tgrp.py records only the clip id
# (video_idx) anyway.
#
# Shard assignment is round-robin over the sorted clip list, so it depends on
# NUM_CHUNKS alone: re-submitting shard 5 re-runs exactly the clips it ran
# before, and individual shards can be resubmitted after a failure.
#
# Each task writes its own JSONL (one shot per line -- the format tgrp.py emits
# and classification.py reads back with load_jsonl), so merging is plain
# concatenation, queued as a dependent job that fires once the array succeeds.
#
# Usage:
#   run.sh                    # 24 shards: submit the array + a dependent merge
#   run.sh 32                 # 32 shards
#   run.sh 24 "3,7,12"        # re-submit only those shards (no merge job)
#   run.sh --merge [n]        # merge the shard outputs by hand (n = NUM_CHUNKS)
#   run.sh --shards [n]       # only build the shard directories, then exit
#
# Overridable via environment:
#   VIDEO_DIR, SAVE_DIR, SAVE_FILE, SHARD_DIR, SHARD_OUT_DIR, SAM2_CHECKPOINT,
#   WORKDIR, LOG_DIR, CONDA_BASE, CONDA_ENV, PARTITION, CONSTRAINT, TIME, MEM, CPUS

set -euo pipefail

VIDEO_DIR="${VIDEO_DIR:-/work/zhongrui/datasets/CMDAM/vid}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-/work/zhongrui/sam2/checkpoints/sam2.1_hiera_large.pt}"

SAVE_DIR="${SAVE_DIR:-/work/zhongrui/datasets/AnimatedAD_data/preds}"
SHARD_OUT_DIR="${SHARD_OUT_DIR:-$SAVE_DIR/tracking_shards}"
SAVE_FILE="${SAVE_FILE:-$SAVE_DIR/tracking_preds.jsonl}"
SHARD_DIR="${SHARD_DIR:-$SAVE_DIR/shards}"

WORKDIR="${WORKDIR:-/work/zhongrui/Animated_AD/character_recognition}"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs/$(date +%m%d)}"

CONDA_BASE="${CONDA_BASE:-/work/zhongrui/anaconda3}"
CONDA_ENV="${CONDA_ENV:-animated_ad}"

PARTITION="${PARTITION:-ddp-2way}"
CONSTRAINT="${CONSTRAINT:-[a6000|a40]}"
TIME="${TIME:-3-00:00:00}"
MEM="${MEM:-80000}"
CPUS="${CPUS:-8}"

# build_shards <num_chunks>
# Deal every clip under VIDEO_DIR round-robin into SHARD_DIR/shard_<i> as
# symlinks. Only symlinks are ever removed or created here; the clips
# themselves are never touched.
build_shards() {
    local num_chunks="$1"

    local videos=()
    mapfile -t videos < <(find "$VIDEO_DIR" -type f \
        \( -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.webm' \) | sort)

    if [ "${#videos[@]}" -eq 0 ]; then
        echo "error: no video files found under ${VIDEO_DIR}" >&2
        exit 1
    fi

    mkdir -p "$SHARD_DIR"
    find "$SHARD_DIR" -mindepth 1 -maxdepth 2 -type l -delete

    local i
    for i in $(seq 0 $((num_chunks - 1))); do
        mkdir -p "$SHARD_DIR/shard_${i}"
    done

    local idx shard video
    for idx in "${!videos[@]}"; do
        video="${videos[$idx]}"
        shard=$((idx % num_chunks))
        ln -sfn "$(readlink -f "$video")" "$SHARD_DIR/shard_${shard}/$(basename "$video")"
    done

    echo "Split ${#videos[@]} clips from ${VIDEO_DIR} into ${num_chunks} shards under ${SHARD_DIR}"
}

# merge_shards <num_chunks>
# Concatenate the per-shard JSONL files in shard order. Every shard that was
# given clips must have produced a file: a silently missing shard would leave a
# gap in the merged predictions that nothing downstream could detect.
merge_shards() {
    local num_chunks="$1"
    local missing=() parts=() i part

    for i in $(seq 0 $((num_chunks - 1))); do
        part="$SHARD_OUT_DIR/tracking_preds_shard_${i}.jsonl"
        if [ -f "$part" ]; then
            parts+=("$part")
        elif [ -d "$SHARD_DIR/shard_${i}" ] && [ -z "$(ls -A "$SHARD_DIR/shard_${i}")" ]; then
            :   # shard was given no clips (num_chunks > number of clips)
        else
            missing+=("$i")
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        echo "error: no output for shard(s): ${missing[*]}" >&2
        echo "       re-run them with: bash ${SCRIPT_PATH} ${num_chunks} \"$(IFS=,; echo "${missing[*]}")\"" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$SAVE_FILE")"
    cat "${parts[@]}" > "$SAVE_FILE"
    echo "Merged ${#parts[@]} shard files -> ${SAVE_FILE} ($(wc -l < "$SAVE_FILE") shots)"
}

# submit_array <num_chunks> <array_spec>
# Submit the TGRP array; task i tracks the clips symlinked into shard_i.
submit_array() {
    local num_chunks="$1" array_spec="$2"

    sbatch --parsable \
        --job-name="tgrp" \
        --array="${array_spec}" \
        --nodes=1 \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --partition="${PARTITION}" \
        --gres=gpu:1 \
        --constraint="${CONSTRAINT}" \
        --requeue \
        --output="${LOG_DIR}/tgrp_%A_%a_output.log" \
        --error="${LOG_DIR}/tgrp_%A_%a_error.log" \
        --chdir "$WORKDIR" \
        --export=ALL \
        --wrap "bash -c '
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
python -u visual_recognition/tgrp.py \
    --sam2_checkpoint ${SAM2_CHECKPOINT} \
    --source_dir ${SHARD_DIR}/shard_\${SLURM_ARRAY_TASK_ID} \
    --save_file ${SHARD_OUT_DIR}/tracking_preds_shard_\${SLURM_ARRAY_TASK_ID}.jsonl
'"
}

# submit_merge <num_chunks> <array_job_id>
# Concatenate the shards once every array task has succeeded.
submit_merge() {
    local num_chunks="$1" dep="$2"

    sbatch --parsable \
        --job-name="tgrp_merge" \
        --dependency="afterok:${dep}" \
        --nodes=1 \
        --cpus-per-task=1 \
        --mem=4000 \
        --time=00:20:00 \
        --partition="${PARTITION}" \
        --output="${LOG_DIR}/tgrp_merge_%j.log" \
        --error="${LOG_DIR}/tgrp_merge_%j.log" \
        --chdir "$WORKDIR" \
        --export=ALL \
        --wrap "bash ${SCRIPT_PATH} --merge ${num_chunks}"
}

# ---- entry point ----

if [ "${1:-}" = "--merge" ]; then
    merge_shards "${2:-24}"
    exit 0
fi

if [ "${1:-}" = "--shards" ]; then
    build_shards "${2:-24}"
    exit 0
fi

# Usage: run.sh [num_chunks] [array_spec]
#   num_chunks: how many shards the clip list is split into (default 24)
#   array_spec: which shard indices to (re)submit, e.g. "0-23" or "1,4,5"
#               (default all: "0-$((num_chunks - 1))")
NUM_CHUNKS="${1:-12}"
ARRAY_SPEC="${2:-0-$((NUM_CHUNKS - 1))}"

[ -d "$VIDEO_DIR" ] || { echo "error: VIDEO_DIR not found: $VIDEO_DIR" >&2; exit 1; }
[ -f "$SAM2_CHECKPOINT" ] || { echo "error: SAM2 checkpoint not found: $SAM2_CHECKPOINT" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$SHARD_OUT_DIR"

build_shards "$NUM_CHUNKS"

ARRAY_JOB_ID=$(submit_array "$NUM_CHUNKS" "$ARRAY_SPEC")
echo "Submitted tgrp array ${ARRAY_JOB_ID} (indices=${ARRAY_SPEC}, num_chunks=${NUM_CHUNKS})"
echo "Clips    <- ${SHARD_DIR}/shard_<i>"
echo "Tracks   -> ${SHARD_OUT_DIR}/tracking_preds_shard_<i>.jsonl"
echo "Logs     -> ${LOG_DIR}/tgrp_${ARRAY_JOB_ID}_<i>_output.log"

# Only chain the merge when the whole array was submitted: after a partial
# re-submission the other shards belong to a different job, so afterok would
# merge as soon as this subset finished and pick up stale or absent outputs.
if [ "$ARRAY_SPEC" = "0-$((NUM_CHUNKS - 1))" ]; then
    MERGE_JOB_ID=$(submit_merge "$NUM_CHUNKS" "$ARRAY_JOB_ID")
    echo "Submitted merge job ${MERGE_JOB_ID} (after ${ARRAY_JOB_ID} succeeds) -> ${SAVE_FILE}"
else
    echo "Partial re-submission: no merge job queued. Merge with:"
    echo "  bash ${SCRIPT_PATH} --merge ${NUM_CHUNKS}"
fi
