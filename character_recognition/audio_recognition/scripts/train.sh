#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."
CONFIG=audio_recognition/config/finetune.yaml

GPUS="${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$GPUS
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

python audio_recognition/finetune.py \
    --config "$CONFIG" \
    --n_gpu "${#GPU_LIST[@]}"
