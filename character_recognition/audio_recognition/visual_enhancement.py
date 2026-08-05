"""
Refine the audio character recognition with the visual recognition results.

For every speech segment that the voice bank was unsure about, the character
tracks that overlap it in time are cut out (`preprocess.crop`) and scored by the
audio-visual synchronisation model: the track whose mouth moves with the speech
is the speaker. The per-segment results are written as JSONL and merged by
`visual_enhancement_postprocessing.py`.

    python audio_recognition/visual_enhancement.py --resume {checkpoint} \
        --audio_prediction_file {...} --track_file {...} --vid_dir {...} \
        --shot_vid_dir {...} --shot_aud_dir {...} --frame_dir {...} \
        --output_dir {...} --save_folder {...}
"""

import argparse
import copy
import json
import os
import subprocess

import torch
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

from data.datasets import DemoDataset
from model.checkpoint import load_checkpoint
from model.lwtnet import LWTNet_temporal_adapter
from model.sync_scorer import DemoEvalTrainerScoreOnly
from preprocess.crop import clean_folder, crop_tracks, extract_audio
from utils.io import get_chunk

# A track has to be at least this confident, and this long, to be considered a
# candidate speaker for a segment.
MIN_TRACK_SCORE = 0.5
MIN_TRACK_FRAMES = 15


def preprocess_tracks(tracks, frame_dir):
    """
    Preprocess a list of track dictionaries into a mapping from (movie_title, clip_idx)
    to a list of processed track information.
    """
    clip_idx_to_tracks = {}

    for track in tracks:
        # Extract metadata
        movie_title = track["movie_title"]
        year = track["year"]
        clip_idx = track["clip_idx"]
        key = (movie_title, clip_idx)
        shot_idx = track["shot_idx"]

        # Construct the path to the shot's frame directory
        shot_dir = os.path.join(frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")

        # Get frame indices from filenames (e.g., "0012.jpg" -> 12)
        frame_ls = [f for f in os.listdir(shot_dir) if f.endswith('.jpg')]
        numbers = [int(f[:-4]) for f in frame_ls]
        start_idx = min(numbers)  # First frame index in this shot

        # Initialize mapping entry if not already present
        if key not in clip_idx_to_tracks:
            clip_idx_to_tracks[key] = []

        # Append processed track info for each classified track
        for track_item in track["classified_tracks"]:
            clip_idx_to_tracks[key].append({
                "shot_idx": shot_idx,
                "track_id": track_item["track_id"],
                "set_idx": track_item["set_idx"],
                "track": track_item["track"],
                "label": track_item["label"],
                "score": track_item["score"],
                "start_idx": start_idx
            })

    return clip_idx_to_tracks


def get_fps(video_path):
    """
    Retrieve the frames per second (FPS) of a video file.
    """
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    return vr.get_avg_fps()


def overlapping_tracks(clip_tracks, start_frame_idx, end_frame_idx):
    """
    The best-scoring track of each character that overlaps [start, end], trimmed
    to the segment.
    """
    candidates = {}

    for track in clip_tracks:
        start_idx = track["start_idx"]
        track_keys = list(track["track"].keys())
        track_start = int(track_keys[0]) + start_idx
        track_end = int(track_keys[-1]) + start_idx

        if start_frame_idx > track_end or end_frame_idx < track_start:
            continue

        # Only consider tracks with high confidence scores
        if track["score"] <= MIN_TRACK_SCORE:
            continue

        final_track = {
            frame_idx: box for frame_idx, box in track["track"].items()
            if (start_frame_idx - start_idx) <= int(frame_idx) <= (end_frame_idx - start_idx)
        }
        final_track = {k: v for k, v in sorted(final_track.items(), key=lambda item: int(item[0]))}

        # Ignore tracks with short durations
        if len(final_track) <= MIN_TRACK_FRAMES:
            continue

        track_ = copy.deepcopy(track)
        track_["track"] = final_track

        label = track_["label"]
        if label not in candidates or track_["score"] > candidates[label]["score"]:
            candidates[label] = track_

    return list(candidates.values())


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load the audio-visual synchronisation model
    model = LWTNet_temporal_adapter().to(device)
    if args.resume:
        load_checkpoint(args.resume, model)
    model.eval()

    trainer = DemoEvalTrainerScoreOnly(model, args)

    # Load the audio-only prediction results
    with open(args.audio_prediction_file, 'r') as infile:
        audio_predictions = json.load(infile)

    # Load the tracks from the visual character recognition
    with open(args.track_file, 'r') as infile:
        all_tracks = json.load(infile)

    clip_idx_to_tracks = preprocess_tracks(all_tracks, args.frame_dir)
    vid_path_list = glob(os.path.join(args.vid_dir, "*", "*"))

    movie_title_clip_idx_list = []
    for movie_title, per_movie_predictions in audio_predictions.items():
        for clip_idx in per_movie_predictions:
            movie_title_clip_idx_list.append((movie_title, clip_idx))

    chunk = get_chunk(movie_title_clip_idx_list, args.num_chunks, args.chunk_idx)

    # Iterate through each clip for visually enhanced audio recognition
    for movie_title, clip_idx in tqdm(chunk, total=len(chunk)):

        clip_predictions = audio_predictions[movie_title][clip_idx]

        # Save visually enhanced results
        clip_save_file = os.path.join(args.save_folder, movie_title, clip_idx + ".jsonl")
        os.makedirs(os.path.dirname(clip_save_file), exist_ok=True)
        clip_save = open(clip_save_file, "w")

        key = (movie_title, clip_idx)
        if key not in clip_idx_to_tracks:
            continue
        clip_tracks = clip_idx_to_tracks[key]

        vid_path = None
        for vid_path_ in vid_path_list:
            if clip_idx in vid_path_:
                vid_path = vid_path_
                break

        if vid_path is None:
            print(f"Video path for clip_idx {clip_idx} not found")
            continue

        year = vid_path.split("/")[-2]
        vid_fps = get_fps(vid_path)

        for clip_pred in tqdm(clip_predictions, desc=f"PROCESSING {movie_title}/{clip_idx}",
                              total=len(clip_predictions)):
            try:
                start_time, end_time = clip_pred["temporal_range"]
                start_frame_idx = int(start_time * vid_fps)
                end_frame_idx = int(end_time * vid_fps)

                # Locate all overlapping tracks
                candidates = overlapping_tracks(clip_tracks, start_frame_idx, end_frame_idx)

                potential_speakers = {}
                for track in tqdm(candidates, total=len(candidates)):
                    shot_idx = track["shot_idx"]
                    label = track["label"]

                    video_path = os.path.join(args.shot_vid_dir, movie_title, year, clip_idx, f"{shot_idx:04d}.mp4")
                    audio_path = video_path.replace(args.shot_vid_dir, args.shot_aud_dir).replace("mp4", "wav")

                    try:
                        extract_audio(video_path, audio_path)
                    except subprocess.CalledProcessError as e:
                        print(f"Skipping {video_path} due to error: {e}")
                        continue

                    # Crop the video segment corresponding to a track: extract the relevant
                    # frames from the shot, resize them consistently across the track, align
                    # with the corresponding audio segment, and save as a cropped video.
                    shot_dir = os.path.join(args.frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")
                    temp_vid_dir = os.path.join("tmp", "vid", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    clean_folder(temp_vid_dir)
                    temp_aud_dir = os.path.join("tmp", "aud", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    clean_folder(temp_aud_dir)
                    output_dir = os.path.join(args.output_dir, movie_title, label)

                    output_path = crop_tracks(track["track"], video_path, audio_path, shot_dir,
                                              temp_vid_dir, temp_aud_dir, output_dir, clip_idx, shot_idx)

                    # Use the audio-visual synchronisation model to determine whether the
                    # character in the track is the one speaking
                    dataset = DemoDataset(video_path=output_path, resize=args.resize,
                                          fps=args.fps, sample_rate=args.sample_rate)
                    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.n_workers)

                    with torch.no_grad():
                        try:
                            scores = trainer.eval(dataloader)
                            average_score = torch.mean(scores)
                        except Exception:
                            continue

                    potential_speakers[label] = {
                        "synchronization_score": average_score.item(),
                        "classification_score": track["score"],
                        "track_id": track["track_id"],
                        "set_idx": track["set_idx"],
                        "shot_idx": track["shot_idx"],
                    }
            except Exception:
                clip_pred["visual_prediction"] = {}
                clip_save.write(json.dumps(clip_pred) + "\n")
                continue

            clip_pred["visual_prediction"] = potential_speakers

            clip_save.write(json.dumps(clip_pred) + "\n")
            clip_save.flush()
        clip_save.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use")
    parser.add_argument('--n_workers', type=int, default=0, help='Num data workers')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--resize', default=540, type=int, help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=25, help='Video input fps')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sampling rate')
    parser.add_argument('--n_negative_samples', type=int, default=0, help='Shift range used for synchronization. E.g. set to 30 from -15 to +15 frame shifts')
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint to resume from")

    parser.add_argument("--audio_prediction_file", type=str, required=True, help="Path to the audio prediction JSON file")
    parser.add_argument("--track_file", type=str, required=True, help="Path to the track JSON file")
    parser.add_argument("--vid_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--shot_vid_dir", type=str, required=True, help="Directory containing shot video files")
    parser.add_argument("--shot_aud_dir", type=str, required=True, help="Directory containing shot audio files")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory containing frame images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output videos")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save the final results")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()
    main(args)
