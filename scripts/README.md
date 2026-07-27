# Pipeline Scripts

Runner scripts for the audio-visual character recognition pipeline described in
[`character_recognition/README.md`](../character_recognition/README.md). They wrap the
same entry points, split the dataset across GPUs where it helps, and merge the results.

All paths and hyper-parameters live in [`config.sh`](config.sh). Either edit it once, or
override any value from the environment:

```shell
GPUS=0,1,2,3 SOURCE_DIR=/data/cmdam/videos bash scripts/tgrp_classification.sh
```

Run the scripts from anywhere, they `cd` into `character_recognition` themselves.

## Visual character recognition

```shell
bash scripts/tgrp_classification.sh    # TGRP + character identification (multi-GPU)
bash scripts/visual_postprocessing.sh  # tracks -> frame predictions -> CMD-AM key frames
```

`tgrp_classification.sh` splits the clips in `$SOURCE_DIR` round-robin into one shard per
GPU in `$GPUS`, runs `tgrp.py` followed by `classification.py` on each shard in parallel,
and merges the per-shard outputs into `$VIS_TRACK_FILE` and `$VIS_CLASSIFIED_FILE`. Shards
are symlink directories under `$WORK_DIR/shards`, the source videos are never modified.
Per-shard logs land in `$WORK_DIR/logs/shard_<id>.log`; a shard that dies can be redone by
re-running with `GPUS` restricted to the ids you want.

## Audio character recognition

```shell
bash scripts/audio_recognition.sh      # match speech segments against the voice bank
bash scripts/visual_enhancement.sh     # refine with the visual results (multi-GPU)
```

`visual_enhancement.sh` uses the `--num_chunks` / `--chunk_idx` support of
`visual_enhancement.py` to spread the clips over `$GPUS`, then merges the per-chunk
`jsonl` results with `visual_enhancement_postprocessing.py` into `$AUD_REFINED_FILE`.

## Evaluation

```shell
bash scripts/eval.sh           # visual + audio
bash scripts/eval.sh visual    # character box mIoU + character name AP
bash scripts/eval.sh audio     # audio recognition AP
```

The audio evaluation picks `$AUD_REFINED_FILE` when it exists and falls back to
`$AUD_PRED_FILE`; set `AUD_EVAL_FILE` to evaluate a specific file, and `EVAL_OSR=1` for the
open-set setting.

## Before the first run

The scripts pass every path through the CLI, but a few things still have to be set up by
hand in the Python entry points:

* `character_recognition/visual_recognition/classification.py` reads
  `args.char_feat_dir` (the appearance bank features) but does not register the flag —
  add `parser.add_argument('--char_feat_dir', ...)` before using `$CHAR_FEAT_DIR`.
* the same file has an empty module constant `VIDEO_IDX_TO_MOVIE_TITLES = ""`, which must
  point at the clip-index-to-movie-title mapping JSON.
* `postprocessing.py` and `select.py` expect the `movie_title` / `year` / `clip_idx` /
  `tracks` / `tracking_boxes` fields, while `tgrp.py` and `classification.py` write
  `video_idx` / `shot_idx` / `track`. `SELECT_INPUT_FILE` in `config.sh` exists so the
  selection step can be pointed at whichever file carries the expected layout.
