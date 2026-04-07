# Subtitling Pipeline

## Overview
We generate character-aware subtitles for animated movies by combining automatic speech recognition (ASR), speaker diarisation, and audio-based character recognition. The pipeline has three stages: (1) transcription with word-level timestamps using WhisperX, (2) speaker diarisation to group speech segments by speaker, and (3) character recognition to assign character names to each segment using the audio character bank.

## Transcription
We use WhisperX to transcribe audio files and obtain word-level timestamps. For English audio, we run in transcribe mode; for non-English audio, we run in translate mode with an additional translation step using M2M100.

To transcribe audio files, run:
```shell
python transcribe.py --root_dir {root_dir} --movie_title_to_videos_file {movie_title_to_videos_file} --save_dir {save_dir}
```

## Audio Character Recognition
After obtaining the transcriptions with speaker diarisation, we assign character names to each speech segment by matching the speaker's voice against the audio character bank. We use a cosine similarity-based classifier with the ECAPA-TDNN speaker encoder.

To run audio character recognition, run:
```shell
python audio_recognition.py --transcription_dir {transcription_dir} --audio_dir {audio_dir} --example_audio_file {example_audio_file} --actor_audio_bank_file {actor_audio_bank_file} --temp_audio_dir {temp_audio_dir} --save_dir {save_dir} --threshold {threshold}
```

`--example_audio_file`: path to the audio example bank JSON containing in-movie voice exemplars. <br>
`--actor_audio_bank_file`: path to the actor audio bank JSON containing interview-based voice exemplars. <br>
`--threshold`: cosine similarity threshold for character recognition (default: 0.45).

## Visual Enhancement
After obtaining the audio character recognition results and the visual character recognition tracks, we refine the audio predictions using audio-visual synchronisation. We use a lip-sync model to determine whether a visually tracked character is the active speaker for each speech segment, and override the audio-only prediction when the visual evidence is stronger.

To run visual enhancement, run:
```shell
python ../../character_recognition/audio_recognition/visual_enhancement.py --resume {resume} --audio_prediction_file {audio_prediction_file} --track_file {track_file} --vid_dir {vid_dir} --shot_vid_dir {shot_vid_dir} --shot_aud_dir {shot_aud_dir} --frame_dir {frame_dir} --output_dir {output_dir} --save_folder {save_folder}
```

Then, to postprocess the visual enhancement results by combining the audio and visual scores, run:
```shell
python ../../character_recognition/audio_recognition/visual_enhancement_postprocessing.py --result_dir {result_dir} --save_file {save_file} --threshold {threshold} --alpha {alpha}
```

`--threshold`: audio score threshold above which the visual prediction is not applied. <br>
`--alpha`: scaling factor for the combined visual synchronisation and classification score.

## Evaluation
We evaluate subtitling with Diarisation Error Rate (DER), which measures the accuracy of both speaker segmentation and character identification.

To evaluate DER, run:
```shell
python eval_der.py --gt_file {gt_file} --pred_file {pred_file}
```

This reports DER for both overlapping and non-overlapping speech regions.
