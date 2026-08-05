# Audio Character Recognition

Identifies which character is speaking, by matching a speech segment against the
voice bank and then, where the voice is ambiguous, checking which character's
mouth actually moves with it.

## Layout

```
audio_recognition/
├── audio_recognition.py                  speech segments -> character, against the voice bank
├── visual_enhancement.py                 refine those predictions with the sync model
├── visual_enhancement_postprocessing.py  merge the per-clip results
├── finetune.py                           fine-tune the sync model (see below)
├── config/                               fine-tuning configs
├── model/                                LWTNet, its losses, checkpoint I/O, inference wrappers
├── data/                                 datasets and audio front-end
├── preprocess/                           building the clips that the above consume
└── utils/                                json/chunking, small net helpers, visualisation
```

Every script is run from `character_recognition/`:

```shell
python audio_recognition/<script>.py ...
```

## Recognition

Match each speech segment against the per-character voice bank:

```shell
python audio_recognition/audio_recognition.py \
    --audio_annotation_file {...} --movie_to_video_file {...} \
    --example_audio_file {...} --actor_audio_bank_file {...} \
    --audio_dir {...} --temp_audio_dir {...} --save_predictions_file {...} --cluster
```

Then refine it with the visual tracks, and merge:

```shell
python audio_recognition/visual_enhancement.py --resume {checkpoint} \
    --audio_prediction_file {...} --track_file {...} --vid_dir {...} \
    --shot_vid_dir {...} --shot_aud_dir {...} --frame_dir {...} \
    --output_dir {...} --save_folder {...}

python audio_recognition/visual_enhancement_postprocessing.py \
    --result_dir {...} --save_file {...} --threshold {...} --alpha {...}
```

`scripts/audio_recognition.sh` and `scripts/visual_enhancement.sh` wrap both,
including sharding the second one across GPUs.

## Preprocessing

The in-movie voice exemplars are cut out of the classified visual tracks:

```shell
python audio_recognition/preprocess/build_voice_examples.py \
    --track_results {classified_tracks.json} --video_dir {shot_vid_dir} \
    --frame_dir {frame_dir} --output_dir {output_dir}
```

`preprocess/generate_synthetic.py` builds the synthetic side-by-side clips (one
talking speaker next to one still frame of another) used to teach the sync model
to localise the speaker rather than score the whole frame.

## Fine-tuning the synchronisation model

`visual_enhancement.py` scores a track with LWTNet plus a temporal adapter. The
released LWTNet is trained on real talking faces, so it is fine-tuned on animated
video before use:

```shell
# adapter-only fine-tune on the animated shots, one process per GPU
python audio_recognition/finetune.py --config audio_recognition/config/finetune_animated.yaml

# override any config key from the command line
python audio_recognition/finetune.py --config audio_recognition/config/finetune_animated.yaml \
    data.movie_title=Minions train.lr=5e-5 train.batch_size=32 wandb.enabled=false

# continue an interrupted run
python audio_recognition/finetune.py --config audio_recognition/config/finetune_animated.yaml \
    checkpoint.resume={checkpoint_dir}/last.pth
```

What it does:

* starts from `model.pretrained`, keeping the initialisation of anything the
  checkpoint does not contain — that is how the temporal adapters, which are not
  in the released LWTNet, get trained from scratch on top of trained encoders;
* trains only `model.trainable` (the projection heads and the adapters by
  default, `[]` for a full fine-tune), with the frozen BatchNorm layers held in
  eval mode so their running statistics do not drift;
* matches each 0.6s video chunk to its audio chunk with a symmetric contrastive
  loss, plus the mean-suppression term that forces the match to come from a
  localised peak (the mouth) rather than from the whole frame;
* validates on a held-out 10% of the clips, writing `best.pth` (lowest
  validation loss), periodic `checkpoint_step*.pth`, and `last.pth`.

All defaults live in `DEFAULT_CONFIG` in `finetune.py`; the config files only
set what differs. An unknown key in a config is an error rather than a silent
no-op.

The resulting `.pth` can be passed straight to `visual_enhancement.py --resume`.

### Data expected

One video file per shot, with the audio next to it as 16 kHz mono wav under
`aud/` instead of `vid/`:

```
{data_folder}/{movie}/{year}/{shot}.mp4    ->  .../aud/{movie}/{year}/{shot}.wav
```

Windows shorter than `data.num_frames` (150 frames = 6s) are skipped, so shots
need to be at least that long to contribute.

Frames are fed to the model in 0..255, in training and at inference alike, which
is the range the released LWTNet expects: upstream AVObjects hands
`frames.astype('float32')` straight to the encoder, and the `bn1` running
statistics stored in `avobjects_loc_sep.pt` match that scale (feeding 0..1
instead lands ~255x below them). Nothing here divides by 255 — an earlier local
copy of the inference path did, which quietly ran the pretrained encoder far
below the activation range it was trained for.

One thing to watch when reusing older weights: checkpoints fine-tuned with this
folder *before* the division was removed were trained on 0..1, so they do not
match the current input range. Fine-tune again from the pretrained checkpoint
rather than resuming one of those.
