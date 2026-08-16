#!/usr/bin/env bash
#
#   $DATA_ROOT/
#     character_table.csv              stage 0  one row per character + provenance
#     character_bank.json              stage 1  {movie: [character names]}
#     movie_title_to_imdbid.json       INPUT    you must supply this (see below)
#     images/raw/<movie>/              stage 0  crawled profile + example images
#     images/filtered/<movie>/         stage 0  one instance crop per kept image
#     voice/interviews/                stage 1  downloaded interview audio
#     voice/cast_information.json      stage 1  character -> actor -> videos
#     voice/transcriptions/            stage 1  whisperx transcripts
#     voice/audio_clips/               stage 1  the interviewee's speech clips
#     voice/audio_features/            stage 1  their embeddings
#     voice/voice_bank.json            stage 1  == the audio character bank
#     appearance/checkpoints/<movie>/  stage 2  finetuned_dinov2_weights.pth
#     appearance/features/<movie>/     stage 3  *.npz == the visual character bank
#     logs/                            all stages
#
# Usage:
#   DATA_ROOT=/data/charbank ./scripts/build_charbank.sh            # all four stages
#   DATA_ROOT=/data/charbank ./scripts/build_charbank.sh --check    # preflight only
#   DATA_ROOT=/data/charbank ./scripts/build_charbank.sh --dry-run  # print the commands
#   ./scripts/build_charbank.sh --stages 2,3                        # just finetune + extract
#   ./scripts/build_charbank.sh --from 1                            # stage 1 onwards
#   GPU_IDS=0,2,3 ./scripts/build_charbank.sh --stages 3
#

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
CB_DIR="$REPO_ROOT/build_character_bank"

DATA_ROOT="${DATA_ROOT:-/path/to/character/bank}"

# ---- stage 0: appearance crawl
SRC_CSV="${SRC_CSV:-$CB_DIR/build_appearance_bank/src.csv}"
RAW_IMG_DIR="${RAW_IMG_DIR:-$DATA_ROOT/images/raw}"
FILTERED_IMG_DIR="${FILTERED_IMG_DIR:-$DATA_ROOT/images/filtered}"
CHARACTER_TABLE="${CHARACTER_TABLE:-$DATA_ROOT/character_table.csv}"
MAX_CHARACTERS="${MAX_CHARACTERS:-10}"
K_RETRIEVAL="${K_RETRIEVAL:-15}"
# 1 = crawl with the Fandom link table (build_csv.py); 0 = link-free crawl
# (build_online.py), for movies with no Fandom entry.
USE_FANDOM="${USE_FANDOM:-1}"
# 1 = cut each crawled image down to one instance crop of the intended
# character. Downstream stages then run on the crops; see IMG_DIR below.
FILTER="${FILTER:-0}"

# ---- stage 1: voice bank
CHARACTER_BANK_JSON="${CHARACTER_BANK_JSON:-$DATA_ROOT/character_bank.json}"
MOVIE_TO_IMDBID="${MOVIE_TO_IMDBID:-$DATA_ROOT/movie_title_to_imdbid.json}"
INTERVIEW_DIR="${INTERVIEW_DIR:-$DATA_ROOT/voice/interviews}"
CAST_INFO_FILE="${CAST_INFO_FILE:-$DATA_ROOT/voice/cast_information.json}"
TRANSCRIPTION_DIR="${TRANSCRIPTION_DIR:-$DATA_ROOT/voice/transcriptions}"
AUDIO_CLIP_DIR="${AUDIO_CLIP_DIR:-$DATA_ROOT/voice/audio_clips}"
AUDIO_FEATURE_DIR="${AUDIO_FEATURE_DIR:-$DATA_ROOT/voice/audio_features}"
VOICE_BANK_FILE="${VOICE_BANK_FILE:-$DATA_ROOT/voice/voice_bank.json}"
COOKIES="${COOKIES:-}"

# ---- stages 2-3: appearance features
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/appearance/checkpoints}"
FEAT_DIR="${FEAT_DIR:-$DATA_ROOT/appearance/features}"
MODEL_SIZE="${MODEL_SIZE:-giant}"
EPOCHS="${EPOCHS:-75}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
SAVE_INTERVAL="${SAVE_INTERVAL:-75}"

# Which images stages 2 and 3 consume. With FILTER=1 they are pre-selected
# instance crops, so feature_extraction.py embeds every one as-is (--embed_all).
# With FILTER=0 they are raw crawled frames, and extraction must instead take
# the anchor path: each character's _0 profile image plus its nearest
# retrievals, cropped with OWLv2 + SAM2 — which is why SAM2 is required there.
if [ "$FILTER" = "1" ]; then
    IMG_DIR="${IMG_DIR:-$FILTERED_IMG_DIR}"
    EMBED_ALL="${EMBED_ALL:-1}"
else
    IMG_DIR="${IMG_DIR:-$RAW_IMG_DIR}"
    EMBED_ALL="${EMBED_ALL:-0}"
fi

# Only needed to white out backgrounds (MASK=1) or for the EMBED_ALL=0 path.
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$REPO_ROOT/checkpoints/sam2.1_hiera_large.pt}"
MASK="${MASK:-0}"

LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs}"
CONDA_ENV="${CONDA_ENV:-animated_ad}"
GPU_IDS="${GPU_IDS:-}"

STAGES="0 1 2 3"
CHECK_ONLY=0
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check)   CHECK_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --stages) STAGES="$(echo "${2:?--stages needs a list, e.g. 0,1}" | tr ',' ' ')"; shift 2 ;;
        --from)   STAGES="$(seq "${2:?--from needs a stage number}" 3 | tr '\n' ' ')"; shift 2 ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 1 ;;
    esac
done

wants() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

log()  { printf '\n\033[1m=== %s\033[0m  %s\n' "$1" "$(date -Is)"; }
note() { printf '    %s\n' "$1"; }
die()  { echo "error: $*" >&2; exit 1; }

# Run a stage, teeing its output to a log file. pipefail makes the python exit
# status win over tee's, so a failure still aborts the run.
run_stage() {
    local name="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        printf '    would run:'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    local logfile="$LOG_DIR/$name.log"
    note "log -> $logfile"
    "$@" 2>&1 | tee "$logfile"
}

activate_conda() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
        return 0
    fi
    local base=""
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base)"
    else
        for c in "$HOME/anaconda3" "$HOME/miniconda3" /work/zhongrui/anaconda3 /opt/conda; do
            if [ -f "$c/etc/profile.d/conda.sh" ]; then
                base="$c"
                break
            fi
        done
    fi
    if [ -z "$base" ] || [ ! -f "$base/etc/profile.d/conda.sh" ]; then
        die "could not locate conda; activate '$CONDA_ENV' yourself, or set
       CONDA_ENV to the env you want (CONDA_ENV=\$CONDA_DEFAULT_ENV to reuse
       the current one)"
    fi
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
}

preflight() {
    local fail=0
    check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=1; fi; }

    echo "Preflight  (stages: $STAGES)"
    check "repo layout               ($CB_DIR)"        "[ -d '$CB_DIR' ]"
    check "data root is writable     ($DATA_ROOT)"     "mkdir -p '$DATA_ROOT' && [ -w '$DATA_ROOT' ]"
    check "python + torch import"                      "python -c 'import torch, PIL, tqdm'"

    if wants 0; then
        if [ "$USE_FANDOM" = "1" ]; then
            check "src.csv                   ($SRC_CSV)" "[ -f '$SRC_CSV' ]"
        fi
        check "build_csv.py present"                    "[ -f '$CB_DIR/build_appearance_bank/build_csv.py' ]"
        if [ "$FILTER" = "1" ]; then
            check "clustering.py present"               "[ -f '$CB_DIR/build_appearance_bank/clustering.py' ]"
        fi
    fi

    if wants 1; then
        # Stage 1 starts from stage 0's character table; only demand it when
        # stage 0 is not part of this run (otherwise it does not exist yet).
        if ! wants 0; then
            check "character table          ($CHARACTER_TABLE)" "[ -f '$CHARACTER_TABLE' ]"
        fi
        # Neither of these can be produced by this pipeline, and stage 1 dies
        # deep inside the crawl without them -- so fail here instead.
        check "movie->imdbid map        ($MOVIE_TO_IMDBID)" "[ -f '$MOVIE_TO_IMDBID' ]"
        check "HF_TOKEN set (gated pyannote models)"        "[ -n \"\${HF_TOKEN:-}\" ]"
        check "yt-dlp available"                           "command -v yt-dlp || python -c 'import yt_dlp'"
        if [ -n "$COOKIES" ]; then
            check "cookies file            ($COOKIES)"  "[ -f '$COOKIES' ]"
        fi
    fi

    if wants 2 || wants 3; then
        check "torch sees a CUDA device"  "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
        check "finetune.py present"       "[ -f '$CB_DIR/feature_extraction/finetune.py' ]"
        check "feature_extraction.py present" "[ -f '$CB_DIR/feature_extraction/feature_extraction.py' ]"
        check "GPU_IDS format"            "[ -z '$GPU_IDS' ] || [[ '$GPU_IDS' =~ ^[0-9]+(,[0-9]+)*$ ]]"
        # Stage 2/3 consume stage 0's output; only demand it when stage 0 is not
        # part of this run (otherwise it does not exist yet, by design).
        if ! wants 0; then
            check "input images              ($IMG_DIR)" "[ -d '$IMG_DIR' ] && [ -n \"\$(ls -A '$IMG_DIR' 2>/dev/null)\" ]"
        fi
    fi
    if { wants 3 && [ "$EMBED_ALL" != "1" ]; } || [ "$MASK" = "1" ]; then
        check "sam2 checkpoint           ($SAM2_CHECKPOINT)" "[ -f '$SAM2_CHECKPOINT' ]"
    fi

    if [ "$fail" -ne 0 ]; then
        die "preflight failed -- fix the FAIL lines above"
    fi
    echo "Preflight passed."
}

stage0_appearance() {
    log "STAGE 0/3  character images"
    note "images -> $RAW_IMG_DIR"
    note "table  -> $CHARACTER_TABLE"
    mkdir -p "$RAW_IMG_DIR" "$(dirname "$CHARACTER_TABLE")"

    if [ "$USE_FANDOM" = "1" ]; then
        run_stage 0a_build_csv \
            python "$CB_DIR/build_appearance_bank/build_csv.py" \
                --src_csv        "$SRC_CSV" \
                --save_dir       "$RAW_IMG_DIR" \
                --save_file      "$CHARACTER_TABLE" \
                --max_characters "$MAX_CHARACTERS" \
                --k_retrieval    "$K_RETRIEVAL"
    else
        run_stage 0a_build_online \
            python "$CB_DIR/build_appearance_bank/build_online.py" \
                --src_csv   "$SRC_CSV" \
                --save_dir  "$RAW_IMG_DIR" \
                --save_file "$CHARACTER_TABLE"
    fi

    if [ "$FILTER" = "1" ]; then
        log "STAGE 0/3  filtering to instance crops"
        note "crops -> $FILTERED_IMG_DIR"
        mkdir -p "$FILTERED_IMG_DIR"
        local mask_args=()
        if [ "$MASK" = "1" ]; then
            mask_args=(--mask --sam2_checkpoint "$SAM2_CHECKPOINT")
        fi
        run_stage 0b_clustering \
            python "$CB_DIR/build_appearance_bank/clustering.py" \
                --img_dir    "$RAW_IMG_DIR" \
                --save_dir   "$FILTERED_IMG_DIR" \
                --model_size "$MODEL_SIZE" \
                "${mask_args[@]}"
    else
        note "FILTER=0 -- keeping the raw crawled images"
    fi
}

stage1_voice() {
    log "STAGE 1/3  voice bank"
    mkdir -p "$INTERVIEW_DIR" "$TRANSCRIPTION_DIR" "$AUDIO_CLIP_DIR" \
             "$AUDIO_FEATURE_DIR" "$(dirname "$VOICE_BANK_FILE")"

    # The character table is one row per character; the crawler wants
    # {movie: [characters]}.
    note "character table -> $CHARACTER_BANK_JSON"
    run_stage 1a_csv_to_json \
        python "$CB_DIR/build_appearance_bank/csv_to_json.py" \
            --character_table "$CHARACTER_TABLE" \
            --save_file       "$CHARACTER_BANK_JSON"

    note "interviews -> $INTERVIEW_DIR"
    local cookie_args=()
    if [ -n "$COOKIES" ]; then
        cookie_args=(--cookies "$COOKIES")
    fi
    run_stage 1b_interview_crawling \
        python "$CB_DIR/build_voice_bank/interview_crawling.py" \
            --save_folder                "$INTERVIEW_DIR" \
            --save_file                  "$CAST_INFO_FILE" \
            --movie_title_to_imdbid_file "$MOVIE_TO_IMDBID" \
            --character_bank_file        "$CHARACTER_BANK_JSON" \
            "${cookie_args[@]}"

    note "voice bank -> $VOICE_BANK_FILE"
    run_stage 1c_voice_clustering \
        python "$CB_DIR/build_voice_bank/clustering.py" \
            --src_dir                "$INTERVIEW_DIR" \
            --cast_information_file  "$CAST_INFO_FILE" \
            --save_transcription_dir "$TRANSCRIPTION_DIR" \
            --audio_clip_dir         "$AUDIO_CLIP_DIR" \
            --audio_feature_dir      "$AUDIO_FEATURE_DIR" \
            --save_file              "$VOICE_BANK_FILE"
}

stage2_finetune() {
    log "STAGE 2/3  test-time finetune DINOv2 per movie"
    note "images  <- $IMG_DIR"
    note "weights -> $CKPT_DIR/<movie>/finetuned_dinov2_weights.pth"
    mkdir -p "$CKPT_DIR"

    local gpu_ids=() num_workers worker_idx gpu logfile pid rc=0
    if [ -n "$GPU_IDS" ]; then
        IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
    else
        gpu_ids=("")
    fi
    num_workers=${#gpu_ids[@]}

    # finetune.py already supports chunking the movie list. Build a temporary
    # input root containing only unfinished movies so reruns resume cleanly.
    local finetune_input="$IMG_DIR" pending_dir="" movie_dir movie
    local pending_movies=()
    if [ "$DRY_RUN" != "1" ]; then
        shopt -s nullglob
        for movie_dir in "$IMG_DIR"/*/; do
            movie="$(basename "$movie_dir")"
            if [ -f "$CKPT_DIR/$movie/finetuned_dinov2_weights.pth" ]; then
                note "skip finetune (complete): $movie"
            elif [ -d "$CKPT_DIR/$movie" ]; then
                die "incomplete checkpoint directory exists: $CKPT_DIR/$movie
       Move or remove it before retrying stage 2."
            else
                pending_movies+=("$movie_dir")
            fi
        done
        shopt -u nullglob

        if [ "${#pending_movies[@]}" -eq 0 ]; then
            note "all movie checkpoints already exist"
            return 0
        fi

        pending_dir="$(mktemp -d "${TMPDIR:-/tmp}/build-charbank-finetune.XXXXXX")"
        for movie_dir in "${pending_movies[@]}"; do
            ln -s "$movie_dir" "$pending_dir/$(basename "$movie_dir")"
        done
        finetune_input="$pending_dir"
    fi

    local pids=()
    for worker_idx in "${!gpu_ids[@]}"; do
        gpu="${gpu_ids[$worker_idx]}"
        logfile="2_finetune"
        if [ -n "$gpu" ]; then
            logfile="${logfile}_gpu_${gpu}"
            run_stage "$logfile" env CUDA_VISIBLE_DEVICES="$gpu" \
                python "$CB_DIR/feature_extraction/finetune.py" \
                    --img_dir       "$finetune_input" \
                    --save_dir      "$CKPT_DIR" \
                    --model_size    "$MODEL_SIZE" \
                    --epochs        "$EPOCHS" \
                    --learning_rate "$LEARNING_RATE" \
                    --temperature   "$TEMPERATURE" \
                    --save_interval "$SAVE_INTERVAL" \
                    --num-chunks    "$num_workers" \
                    --chunk-idx     "$worker_idx" &
        else
            run_stage "$logfile" \
                python "$CB_DIR/feature_extraction/finetune.py" \
                    --img_dir       "$finetune_input" \
                    --save_dir      "$CKPT_DIR" \
                    --model_size    "$MODEL_SIZE" \
                    --epochs        "$EPOCHS" \
                    --learning_rate "$LEARNING_RATE" \
                    --temperature   "$TEMPERATURE" \
                    --save_interval "$SAVE_INTERVAL" \
                    --num-chunks    "$num_workers" \
                    --chunk-idx     "$worker_idx" &
        fi
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then rc=1; fi
    done

    if [ -n "$pending_dir" ]; then
        # The directory contains only symlinks created immediately above.
        rm -rf -- "$pending_dir"
    fi
    return "$rc"
}

extract_worker() {
    local worker_idx="$1" num_workers="$2" gpu="$3"
    local movie_dirs=() movie_dir movie movie_ckpt movie_feat index=0
    local extract_args=() existing_features=()

    shopt -s nullglob
    movie_dirs=("$IMG_DIR"/*/)
    for movie_dir in "${movie_dirs[@]}"; do
        if [ $((index % num_workers)) -ne "$worker_idx" ]; then
            index=$((index + 1))
            continue
        fi
        index=$((index + 1))

        movie="$(basename "$movie_dir")"
        movie_ckpt="$CKPT_DIR/$movie/finetuned_dinov2_weights.pth"
        movie_feat="$FEAT_DIR/$movie"
        # In a full dry run stage 2 has not produced this file yet; still show
        # the stage-3 command that would consume it.
        if [ "$DRY_RUN" != "1" ] && [ ! -f "$movie_ckpt" ]; then
            echo "error: missing checkpoint for '$movie': $movie_ckpt" >&2
            return 1
        fi

        existing_features=("$movie_feat"/*.npz)
        if [ "${#existing_features[@]}" -gt 0 ]; then
            note "skip extraction (complete): $movie"
            continue
        fi

        extract_args=(
            python "$CB_DIR/feature_extraction/feature_extraction.py"
            --img_dir "$movie_dir"
            --save_folder "$movie_feat"
            --pretrained_weights "$movie_ckpt"
            --model_size "$MODEL_SIZE"
        )
        if [ "$EMBED_ALL" = "1" ]; then
            extract_args+=(--embed_all)
        else
            extract_args+=(--box)
            if [ "$MASK" = "1" ]; then
                extract_args+=(--mask --sam2_checkpoint "$SAM2_CHECKPOINT")
            fi
        fi

        if [ -n "$gpu" ]; then
            run_stage "3_extract_$movie" env CUDA_VISIBLE_DEVICES="$gpu" \
                "${extract_args[@]}" || return 1
        else
            run_stage "3_extract_$movie" "${extract_args[@]}" || return 1
        fi
    done
    shopt -u nullglob
}

stage3_extract() {
    log "STAGE 3/3  extract appearance features"
    note "images   <- $IMG_DIR"
    note "features -> $FEAT_DIR/<movie>/*.npz"
    mkdir -p "$FEAT_DIR"

    local gpu_ids=() num_workers worker_idx gpu pid rc=0
    if [ -n "$GPU_IDS" ]; then
        IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
    else
        gpu_ids=("")
    fi
    num_workers=${#gpu_ids[@]}

    local pids=()
    for worker_idx in "${!gpu_ids[@]}"; do
        gpu="${gpu_ids[$worker_idx]}"
        extract_worker "$worker_idx" "$num_workers" "$gpu" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then rc=1; fi
    done
    return "$rc"
}

if [ "$DATA_ROOT" = "/path/to/character/bank" ]; then
    die "DATA_ROOT is still the placeholder. Set it to the folder the whole
       character bank should live in, e.g.
           DATA_ROOT=/data/character_bank $0"
fi

cd "$REPO_ROOT"
activate_conda
mkdir -p "$LOG_DIR"

echo "Character bank build"
echo "  repo      $REPO_ROOT"
echo "  data root $DATA_ROOT"
echo "  stages    $STAGES"
echo "  images    $IMG_DIR  (FILTER=$FILTER, EMBED_ALL=$EMBED_ALL)"

preflight
if [ "$CHECK_ONLY" -eq 1 ]; then exit 0; fi
if [ "$DRY_RUN" = "1" ]; then echo; echo "DRY RUN -- nothing will be executed."; fi

started=$(date +%s)
if wants 0; then stage0_appearance; fi
if wants 1; then stage1_voice; fi
if wants 2; then stage2_finetune; fi
if wants 3; then stage3_extract; fi

log "DONE  ($(( ($(date +%s) - started) / 60 )) min)"
echo "  visual bank  $FEAT_DIR/<movie>/*.npz"
echo "  voice bank   $VOICE_BANK_FILE"
echo
echo "Point the recognition pipeline at them with:"
echo "  CHAR_FEAT_DIR=$FEAT_DIR"
echo "  DINOV2_CKPT_DIR=$CKPT_DIR"
echo "  (see scripts/config.sh)"
