#!/usr/bin/env bash
# Audio character recognition against the voice bank.
#
#   bash scripts/audio_recognition.sh
#   USE_CLUSTER=0 bash scripts/audio_recognition.sh   # skip speaker clustering
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

require_path "$AUDIO_ANNOTATION_FILE" "audio annotation file (AUDIO_ANNOTATION_FILE)"
require_path "$MOVIE_TO_VIDEO_FILE" "movie-to-video mapping (MOVIE_TO_VIDEO_FILE)"
require_path "$EXAMPLE_AUDIO_FILE" "example audio bank (EXAMPLE_AUDIO_FILE)"
require_path "$ACTOR_AUDIO_BANK_FILE" "actor audio bank (ACTOR_AUDIO_BANK_FILE)"
require_path "$AUDIO_DIR" "animated audio directory (AUDIO_DIR)"

mkdir -p "$TEMP_AUDIO_DIR" "$OUTPUT_DIR"
cd "$CR_DIR"

CLUSTER_ARGS=()
if [ "$USE_CLUSTER" = "1" ]; then
    CLUSTER_ARGS=(--cluster)
fi

python audio_recognition/audio_recognition.py \
    --audio_annotation_file "$AUDIO_ANNOTATION_FILE" \
    --movie_to_video_file "$MOVIE_TO_VIDEO_FILE" \
    --example_audio_file "$EXAMPLE_AUDIO_FILE" \
    --actor_audio_bank_file "$ACTOR_AUDIO_BANK_FILE" \
    --audio_dir "$AUDIO_DIR" \
    --temp_audio_dir "$TEMP_AUDIO_DIR" \
    --save_predictions_file "$AUD_PRED_FILE" \
    "${CLUSTER_ARGS[@]}"

echo "Audio predictions:    ${AUD_PRED_FILE}"
echo "Next: bash scripts/visual_enhancement.sh (optional refinement) or bash scripts/eval.sh"
