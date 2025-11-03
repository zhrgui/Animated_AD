# Audio-Visual Character Recognition

## Overview
We use our automatically constructed audio-visual character bank to enable audio-visual recognition of the animated characters. There are two parts in it, which is visual character recognition and audio character recognition. We also use our visual character recognition results to refine our audio character recognition.

## Visual Character Recognition
There are two stages in our pipeline for visual character recognition, which are Track-Guided Region Proposal (TGRP) and character identification.

To run Track-Guided Region Proposal for region proposal, run:
```shell:
python visual_recognition/tgrp.py --source_dir {source_dir} --save_frame_dir {save_frame_dir} --save_file {save_file}
```

Then, to identify the character in these proposed regions, run:
```shell:
python visual_recognition/classification.py --track_file {track_file} --frame_dir {frame_dir} --save_file {save_file} --mask
```

Alternatively, we provide scripts to split the dataset and run on more than one GPUs simultaneously.
```shell:
cd ..
bash scripts/tgrp_classification.sh
```

After obtaining the classified tracks, we convert them into frame results by running:
```shell:
python visual_recognition/postprocessing.py --source_dir {source_dir} --track_file {track_file} --save_file {save_file}
```

To prepare for evaluation on the CMDAM subset, run:
```shell:
python visual_recognition/select.py --annotation_file {annotation_file} --track_predictions_file {track_predictions_file} --save_path {save_path}
```
This will select the predictions for frames with ground-truth boxes to evaluate in the MovieNet style.

## Audio Character Recognition

For audio character recognition, run:
```shell
python audio_recognition/audio_recognition.py --audio_annotation_file {audio_annotation_file} --movie_to_video_file {movie_to_video_file} --example_audio_file {example_audio_file} --actor_audio_bank_file {actor_audio_bank_file} --audio_dir {audio_dir} --temp_audio_dir {temp_audio_dir} --save_predictions_file {save_predictions_file} --cluster
```

After getting the results from visual character recognition, the audio character recognition can be further refined with them. For visual enhancement, run:
```shell
python audio_recognition/visual_enhancement.py --resume {resume} --audio_prediction_file {audio_prediction_file} --track_file {track_file} --vid_dir {vid_dir} --shot_vid_dir {shot_vid_dir} --shot_aud_dir {shot_aud_dir} --frame_dir {frame_dir} --output_dir {output_dir} --save_folder {save_folder}

python audio_recognition/visual_enhancement_postprocessing.py.py --result_dir {result_dir} --save_file {save_file} --threshold {threshold} --alpha {alpha}
```

## Evaluation

We provide the evaluation scripts for both visual and audio character recognition.

We evaluate the visual character recognition with two metrics. To evaluate the character box mIoU, run:
```shell
python eval/eval_box.py --prediction_file {prediction_file} --annotation_file ../resources/cmdam_boxes.json
```

To evaluate the character name AP in the MovieNet style, run:
```shell
python eval/eval_name.py --prediction_file {prediction_file} --annotation_file ../resources/cmdam_boxes.json
```

To evaluate audio recognition on ground-truth time segments, run:
```shell
python eval/eval_asr.py --prediction_file {prediction_file}
```
