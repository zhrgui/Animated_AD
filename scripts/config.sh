#!/usr/bin/env bash
# Shared configuration for the audio-visual character recognition pipeline
# (see character_recognition/README.md).
#
# This file is sourced by the other scripts in this folder, it is not meant to
# be run on its own. Every value below can be overridden from the environment:
#
#   GPUS=0,1,2,3 SOURCE_DIR=/data/cmdam/videos bash scripts/tgrp_classification.sh

# ----------------------------------------------------------------- repo layout
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
CR_DIR="$REPO_ROOT/character_recognition"

# -------------------------------------------------------------------- hardware
# Comma-separated GPU ids. One worker process is launched per id, and the
# dataset is split evenly across the workers.
GPUS="${GPUS:-0}"

# ----------------------------------------------------------------- checkpoints
# SAM2 checkpoint (used by both TGRP and classification).
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$REPO_ROOT/checkpoints/sam2.1_hiera_large.pt}"
# Test-time finetuned DINOv2 weights, one sub-folder per movie:
#   $DINOV2_CKPT_DIR/<movie_title>/finetuned_dinov2_weights.pth
DINOV2_CKPT_DIR="${DINOV2_CKPT_DIR:-$REPO_ROOT/checkpoints/dinov2_finetuned}"
DINOV2_MODEL_SIZE="${DINOV2_MODEL_SIZE:-giant}"
# Appearance bank features, one sub-folder per movie, one .npz per character.
CHAR_FEAT_DIR="${CHAR_FEAT_DIR:-$REPO_ROOT/character_bank/appearance_features}"
# LWTNet checkpoint used by the visual enhancement of audio recognition.
SYNC_CHECKPOINT="${SYNC_CHECKPOINT:-$REPO_ROOT/checkpoints/lwtnet.pth}"

# ------------------------------------------------------------------- video i/o
# Directory of source video clips (one file per clip, named <clip_idx>.mp4).
SOURCE_DIR="${SOURCE_DIR:-$REPO_ROOT/data/videos}"
# Frames extracted per shot by TGRP: $FRAME_DIR/<clip_idx>/<shot_idx>/<frame_idx>.jpg
FRAME_DIR="${FRAME_DIR:-$REPO_ROOT/data/frames}"
# Scratch space for dataset shards, per-shard outputs and logs.
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work}"
# Final results.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs}"

# --------------------------------------------------- visual recognition output
VIS_TRACK_FILE="${VIS_TRACK_FILE:-$OUTPUT_DIR/tracks.jsonl}"                       # TGRP region proposals
VIS_CLASSIFIED_FILE="${VIS_CLASSIFIED_FILE:-$OUTPUT_DIR/classified_tracks.json}"   # + character labels
VIS_FRAME_PRED_FILE="${VIS_FRAME_PRED_FILE:-$OUTPUT_DIR/frame_predictions.json}"   # tracks -> per-frame boxes
VIS_SELECTED_FILE="${VIS_SELECTED_FILE:-$OUTPUT_DIR/cmdam_predictions.json}"       # MovieNet-style eval input
# Crop the background out with SAM2 masks before classification (--mask).
USE_MASK="${USE_MASK:-1}"
# Input of visual_recognition/select_subset.py, i.e. the per-shot predictions to be
# restricted to the annotated key frames.
SELECT_INPUT_FILE="${SELECT_INPUT_FILE:-$VIS_CLASSIFIED_FILE}"

# ---------------------------------------------------------- audio recognition
AUDIO_ANNOTATION_FILE="${AUDIO_ANNOTATION_FILE:-$REPO_ROOT/data/audio/audio_annotations.json}"
MOVIE_TO_VIDEO_FILE="${MOVIE_TO_VIDEO_FILE:-$REPO_ROOT/data/audio/movie_to_video.json}"
EXAMPLE_AUDIO_FILE="${EXAMPLE_AUDIO_FILE:-$REPO_ROOT/data/audio/example_audio_bank.json}"
ACTOR_AUDIO_BANK_FILE="${ACTOR_AUDIO_BANK_FILE:-$REPO_ROOT/data/audio/actor_audio_bank.json}"
AUDIO_DIR="${AUDIO_DIR:-$REPO_ROOT/data/audio/clips}"
TEMP_AUDIO_DIR="${TEMP_AUDIO_DIR:-$WORK_DIR/temp_audio}"
AUD_PRED_FILE="${AUD_PRED_FILE:-$OUTPUT_DIR/audio_predictions.json}"
# Cluster the speech segments before matching against the voice bank (--cluster).
USE_CLUSTER="${USE_CLUSTER:-1}"

# --------------------------------------- visual enhancement of audio results
VID_DIR="${VID_DIR:-$SOURCE_DIR}"                       # full clips
SHOT_VID_DIR="${SHOT_VID_DIR:-$WORK_DIR/shot_videos}"   # per-shot video crops
SHOT_AUD_DIR="${SHOT_AUD_DIR:-$WORK_DIR/shot_audio}"    # per-shot audio crops
VE_OUTPUT_DIR="${VE_OUTPUT_DIR:-$WORK_DIR/enhancement_videos}"   # per-track videos
VE_RESULT_DIR="${VE_RESULT_DIR:-$WORK_DIR/enhancement_results}"  # <movie>/<clip>.jsonl
AUD_REFINED_FILE="${AUD_REFINED_FILE:-$OUTPUT_DIR/audio_predictions_refined.json}"
VE_THRESHOLD="${VE_THRESHOLD:-0.15}"   # audio score below which the visual result is trusted
VE_ALPHA="${VE_ALPHA:-1.0}"            # weight of the visual score

# -------------------------------------------------------------------- evaluation
ANNOTATION_FILE="${ANNOTATION_FILE:-$REPO_ROOT/resources/cmdam_boxes.json}"
EVAL_NUM_SHOT="${EVAL_NUM_SHOT:-3}"
# Evaluate audio recognition in the open-set setting (--osr).
EVAL_OSR="${EVAL_OSR:-0}"

# --------------------------------------------------------------------- helpers
# Abort early with a readable message instead of a stack trace 20 minutes in.
require_path() {
    local path="$1" description="$2"
    if [ ! -e "$path" ]; then
        echo "error: ${description} not found: ${path}" >&2
        echo "       set it in scripts/config.sh or pass it in the environment." >&2
        exit 1
    fi
}

# Parse $GPUS into the GPU_LIST array.
read_gpus() {
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
    if [ "${#GPU_LIST[@]}" -eq 0 ]; then
        echo "error: GPUS is empty" >&2
        exit 1
    fi
}
