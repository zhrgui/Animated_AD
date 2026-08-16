# Audio-Visual Character Bank Construction

## Overview
In the audio-visual character bank, there are image exemplars of characters' appearances and audio exemplar clips of characters' voices. We provide an automatic pipeline for crawling character profile images from [Fandom](https://www.fandom.com/) and additional exemplar images from the web. For audio character bank, we search the actor names on YouTube and crawl the resulted interview videos. We then leverage some postprocessing techniques to extract the voice segments of the queried actors.

<p align="center">
  <img src="pipeline.png"  width="800"/>
</p>

## Automatic Pipeline
Run the complete character-bank pipeline from the repository root with
[`scripts/build_charbank.sh`](../scripts/build_charbank.sh):

```shell
DATA_ROOT=/data/character_bank ./scripts/build_charbank.sh
```

`DATA_ROOT` is where the crawled images, voice data, checkpoints, extracted
features, and logs will be written. Before running the full build, provide
`$DATA_ROOT/movie_title_to_imdbid.json` and export `HF_TOKEN` for the gated
Pyannote models. You can validate the environment or inspect the commands
without running them:

```shell
DATA_ROOT=/data/character_bank ./scripts/build_charbank.sh --check
DATA_ROOT=/data/character_bank ./scripts/build_charbank.sh --dry-run
```

Use `--stages 0,1`, for example, to run selected stages, or `--from 2` to run
stage 2 onward. The main configuration options are environment variables; see
the defaults at the top of the script. In particular, set `FILTER=1` to run
the optional image-filtering step before finetuning and feature extraction,
and set `GPU_IDS=0,2,3` to select GPUs for those stages.


## Visual Character Bank Construction
To build the visual character bank, we provide a CSV file including links to the Fandom pages containing the profile image of each character for all movies in the CMD-AM dataset. To crawl these profile images and save to a local directory, run:

```shell
python build_character_bank/build_appearance_bank/build_csv.py \
  --src_csv build_character_bank/build_appearance_bank/src.csv \
  --save_dir {raw_img_dir} \
  --save_file {character_table} \
  --max_characters 10 \
  --k_retrieval 15
```

Also, this will crawl `--k_retrieval` (15 by default) example images per character and save them to the same directory, in order to populate the character image examples. These come from the character's own wiki gallery first and from Bing Images second, with near-duplicates dropped across both sources. Within a movie folder, the profile image is saved as `{character}_0.png` and the examples as `{character}_1.png ... {character}_k.png`; the rest of the pipeline relies on that naming. Every character found is saved to `{save_file}` as a table, one row per character, recording the wiki page its profile image came from and how confident that anchor is (`confident`, `low` or `missing`). Pass a path ending in `.csv` for a CSV table, anything else for JSON rows.

Additionally, we provide an automatic pipeline for crawling profile images without Fandom links. However, this sometimes suffer from 404 errors, as the format of the links varies for different movies.

```shell
python build_character_bank/build_appearance_bank/build_online.py \
  --src_csv build_character_bank/build_appearance_bank/src.csv \
  --save_dir {raw_img_dir} \
  --save_file {character_table}
```

### Filtering the retrieved images (Optional)
Crawled images are noisy: they contain co-starring characters, crowd shots, body-part close-ups and blurry frames, and the character we asked for is usually only part of the frame. The clustering stage cuts each retrieved image down to at most one instance crop of the intended character, and throws the image away when no crop is convincing:

```shell
python build_character_bank/build_appearance_bank/clustering.py \
  --img_dir {raw_img_dir} \
  --save_dir {filtered_img_dir} \
  --model_size giant
```

Each retrieved image is run through OWLv2 to propose character boxes, and DINOv2 embeds each box crop. Selection then takes one of two paths per character:

- **With a profile image** (`{character}_0.png`), the instance most similar to it is kept per image. All of a movie's profile images are embedded up front, and an instance is discarded when some *other* character in the same movie explains it better (`--disambig_margin`), which stops a co-star's crop from being credited to the wrong bank entry. Because wiki portraits are often stylistically far from in-film renders, a second round rescores every instance against per-character prototypes — the profile image averaged with the confident round-1 picks — so look-alikes are separated using the film's own style. A character whose every candidate lost the cross-character vote is retried without the veto, keeping a single image, and is flagged `[rescued]` in the log for manual review.
- **Without a profile image**, the instances pooled over all of the character's retrieved images are clustered (agglomerative, cosine distance, `--cluster_threshold`). Clusters are ranked by how many *distinct* images they cover rather than by raw instance count, since the intended character is the one that recurs, and the winning cluster's centroid then plays the profile image's role.

Both paths share a quality-first keep rule with a quota: images pass outright above `--sim_threshold`, and the best `--min_keep` are kept anyway as long as they clear `--floor_threshold`, so sparse characters still end up with examples. Detection-side filters drop group boxes containing two or more other detections, boxes smaller than `--min_box_size`, boxes scoring below `--rel_box_ratio` of the image's best box, and crops blurrier than `--min_sharpness`. Pass `--mask` (with `--sam2_checkpoint`) to white out the background using SAM2 masks; this is off by default because SAM masks are sometimes incomplete. `--movie` restricts the run to one movie folder, and movies whose output folder is non-empty are skipped unless `--overwrite` is given.

Crops are written to `--save_dir/{movie}/` under their original filenames, and this filtered directory is what the feature extraction below should be pointed at.

After crawling the exemplar images, we finetune the DINOv2 contrastively on each movie and extract the visual features from these images for later feature matching.

To finetune DINOv2 at test time for each movie, run:
```shell
python build_character_bank/feature_extraction/finetune.py \
  --img_dir {img_dir} \
  --save_dir {checkpoint_dir} \
  --model_size giant \
  --epochs 75 \
  --learning_rate 6e-4 \
  --temperature 0.07 \
  --save_interval 75
```

Then, the visual character bank can be derived using the finetuned DINOv2 features:
```shell
python build_character_bank/feature_extraction/feature_extraction.py \
  --img_dir {movie_img_dir} \
  --save_folder {movie_feature_dir} \
  --pretrained_weights {movie_checkpoint}/finetuned_dinov2_weights.pth \
  --model_size giant \
  --box
```

When `{movie_img_dir}` contains the pre-filtered instance crops produced with
`FILTER=1`, also pass `--embed_all`.

## Audio Character Bank Construction
Our audio character bank has two sources of audio exemplars, which are actor interviews and in-movie voice exemplars. We start to build it with online crawled actor interviews.

This stage reads the character names as a `{movie title: [character names]}` JSON, whereas `build_csv.py` and `build_online.py` write one row per character carrying the provenance of its profile image. Rather than rebuilding the character lists by hand, collapse the character table written above into that shape:

```shell
python build_character_bank/build_appearance_bank/csv_to_json.py \
  --character_table {character_table} \
  --save_file {character_bank_file}
```

`{character_table}` is the `--save_file` CSV from either builder. Row order is preserved, so characters stay in the order the builder ranked them (IMDb billing order, then wiki article length) and the most prominent characters are crawled first. The command prints the movie and character counts it wrote.

Using that JSON file, we load the list of character names for each movie. We then crawl the mapping of character names to cast names on IMDb and get a list of actor names. We then query YouTube and download the most relevant videos. This can be done by:

```shell
python build_character_bank/build_voice_bank/interview_crawling.py \
  --save_folder {interview_dir} \
  --save_file {cast_information_file} \
  --movie_title_to_imdbid_file {movie_title_to_imdbid_file} \
  --character_bank_file {character_bank_file}
```

After downloading the interview videos, we adopt a straight-forward clustering algorithm to extract the voice clips of our target actors. We leverage a prior that in the interviews, the largest cluster corresponds to the interviewees.

```shell
python build_character_bank/build_voice_bank/clustering.py \
  --src_dir {interview_dir} \
  --cast_information_file {cast_information_file} \
  --save_transcription_dir {transcription_dir} \
  --audio_clip_dir {audio_clip_dir} \
  --audio_feature_dir {audio_feature_dir} \
  --save_file {voice_bank_file}
```

This will extract the audio features of the resulted audio clips and save for later audio feature matching.
