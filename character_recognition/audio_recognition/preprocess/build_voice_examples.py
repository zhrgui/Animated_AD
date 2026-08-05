"""
Build the in-movie voice exemplars of the audio character bank.

Takes the classified tracks from visual character recognition, keeps the
highest-scoring track of each character in each shot, and cuts that track out as
a video + audio clip. Those clips are what `audio_recognition.py` later encodes
into the per-character voice bank.

    python audio_recognition/preprocess/build_voice_examples.py \
        --track_results {classified_tracks.json} --video_dir {shot_vid_dir} \
        --frame_dir {frame_dir} --output_dir {output_dir}
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Runnable as a script from anywhere: put the package root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.crop import crop_tracks, extract_audio  # noqa: E402
from utils.io import load_json, save_json  # noqa: E402

# Characters that appear as a crowd rather than as one identity, so every track
# is worth keeping instead of only the highest scoring one.
CROWD_CHARACTERS = ("Minions",)

# Fall back to lower-scoring tracks until a character has at least this many
# exemplars.
MIN_EXAMPLES_PER_CHARACTER = 10


def filter_repeating_characters(tracks):
    """
    Filter out duplicate character tracks by keeping only the highest-scoring track per character,
    except for the crowd characters where all tracks are retained.
    """
    char_tracks = {}
    allowed_tracks = []

    for track in tracks:
        if track["label"] in CROWD_CHARACTERS:
            allowed_tracks.append(track)
        if track["label"] not in char_tracks.keys():
            char_tracks[track["label"]] = track
        else:
            if track["score"] > char_tracks[track["label"]]["score"]:
                char_tracks[track["label"]] = track

    return list(char_tracks.values()) + allowed_tracks


def save_track_instance(track_instance, label, key_to_tracks, args):
    """
    Cut the track of `label` in the shot identified by `track_instance` out into
    its own clip, and return where it was written.
    """
    movie_title, year, clip_idx, shot_idx = track_instance[0]

    instance_track = None
    for track in key_to_tracks[track_instance[0]]:
        if track["label"] == label:
            instance_track = track["track"]

    video_path = os.path.join(args.video_dir, movie_title, year, clip_idx, f"{shot_idx:04d}.mp4")
    audio_path = video_path.replace("vid", "aud").replace("mp4", "wav")
    extract_audio(video_path, audio_path)

    shot_dir = os.path.join(args.frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")
    temp_vid_dir = os.path.join(args.temp_dir, "vid", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
    temp_aud_dir = os.path.join(args.temp_dir, "aud", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
    output_dir = os.path.join(args.output_dir, movie_title, label)

    return crop_tracks(instance_track, video_path, audio_path, shot_dir,
                       temp_vid_dir, temp_aud_dir, output_dir, clip_idx, shot_idx)


def main(args):
    # Load the detected tracks from visual character recognition
    track_file = load_json(args.track_results)

    # Initialize retrieved tracks for potential in-movie audio exemplars
    retrieved_character_information = defaultdict(dict)
    key_to_tracks = defaultdict(dict)

    for shot_annotations in track_file:
        movie_title = shot_annotations["movie_title"]

        if args.movie_title and movie_title != args.movie_title:
            continue

        key = (movie_title, shot_annotations["year"],
               shot_annotations["clip_idx"], shot_annotations["shot_idx"])

        filtered_tracks = filter_repeating_characters(shot_annotations["classified_tracks"])
        key_to_tracks[key] = filtered_tracks

        for track in filtered_tracks:
            label = track["label"]
            score = track["score"]

            if key not in retrieved_character_information[label] or score > retrieved_character_information[label][key]:
                retrieved_character_information[label][key] = score

    # Rank the shots of every character by how confident the visual recognition was
    ranked_by_label = {
        label: sorted(shots.items(), key=lambda x: x[1], reverse=True)
        for label, shots in retrieved_character_information.items()
    }

    instance_confidence = {}
    for label, track_instances in ranked_by_label.items():
        instance_confidence[label] = {}
        saved_track_instances = []

        # First pass: every track the classifier was confident about.
        for track_instance in track_instances:
            if track_instance[1] <= args.thres:
                continue
            try:
                output_path = save_track_instance(track_instance, label, key_to_tracks, args)
            except Exception:
                continue
            instance_confidence[label][output_path] = track_instance[1]
            saved_track_instances.append(track_instance)

        # Second pass: top up with the best of the rest, so that a character with
        # few confident tracks still gets a usable number of exemplars.
        for track_instance in track_instances:
            if len(saved_track_instances) >= MIN_EXAMPLES_PER_CHARACTER:
                break
            if track_instance in saved_track_instances:
                continue
            try:
                output_path = save_track_instance(track_instance, label, key_to_tracks, args)
            except Exception:
                continue
            instance_confidence[label][output_path] = track_instance[1]
            saved_track_instances.append(track_instance)

        print(f"{label}: {len(saved_track_instances)} exemplars")

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(instance_confidence, os.path.join(args.output_dir, "instance_confidence.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_results', required=True, type=str, help='Path to track results JSON')
    parser.add_argument('--video_dir', required=True, type=str, help='Directory containing shot video files')
    parser.add_argument('--frame_dir', required=True, type=str, help='Directory containing frame images')
    parser.add_argument('--output_dir', required=True, type=str, help='Output directory for instance videos')
    parser.add_argument('--temp_dir', default='tmp', type=str, help='Scratch directory for the cropped frames')
    parser.add_argument('--movie_title', default=None, type=str)
    parser.add_argument('--thres', default=0.6, type=float)

    args = parser.parse_args()

    main(args)
