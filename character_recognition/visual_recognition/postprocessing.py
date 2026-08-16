import argparse
import json
import os

from decord import VideoReader, cpu


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def index_videos(source_dir):
    """Return a mapping from clip id to video path below ``source_dir``."""
    videos = {}
    for root, dirnames, filenames in os.walk(source_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            clip_idx, extension = os.path.splitext(filename)
            if extension.lower() not in VIDEO_EXTENSIONS:
                continue

            video_path = os.path.join(root, filename)
            if clip_idx in videos:
                raise ValueError(
                    f"duplicate video id {clip_idx!r}: {videos[clip_idx]} and {video_path}"
                )
            videos[clip_idx] = video_path

    if not videos:
        raise ValueError(f"no video files found under {source_dir}")
    return videos


def empty_prediction():
    return {"bbox_ls": [], "labels": [], "scores": []}


def main():
    with open(args.track_file, "r") as infile:
        tracks = json.load(infile)

    video_paths = index_videos(args.source_dir)
    results = {}
    clip_to_movie = {}

    for track in tracks:
        movie_title = track["movie_title"]
        clip_idx = track.get("clip_idx", track.get("video_idx"))
        if clip_idx is None:
            raise KeyError("track record has neither 'clip_idx' nor 'video_idx'")
        if clip_idx not in video_paths:
            raise FileNotFoundError(
                f"video for clip {clip_idx!r} was not found under {args.source_dir}"
            )

        previous_movie = clip_to_movie.setdefault(clip_idx, movie_title)
        if previous_movie != movie_title:
            raise ValueError(
                f"clip {clip_idx!r} is assigned to both {previous_movie!r} and {movie_title!r}"
            )

        clip_frames = results.setdefault(movie_title, {}).setdefault(clip_idx, {})

        # TGRP records shot-local frame indices in each object track and the
        # shot's absolute first video frame in start_idx. This replaces looking
        # at the old extracted-frame folder to recover the same offset.
        if "start_idx" not in track:
            raise KeyError(
                f"track for clip {clip_idx!r}, shot {track.get('shot_idx')!r} "
                "has no 'start_idx'"
            )
        shot_start_idx = int(track["start_idx"])

        track_instances = track.get("tracks", track.get("track"))
        if track_instances is None:
            raise KeyError(
                f"track for clip {clip_idx!r}, shot {track.get('shot_idx')!r} "
                "has neither 'tracks' nor 'track'"
            )

        for track_instance in track_instances:
            track_sequence = track_instance["track"]
            track_label = track_instance["label"]
            track_score = track_instance["score"]

            for frame_idx, box in track_sequence.items():
                global_frame_idx = str(shot_start_idx + int(frame_idx))
                prediction = clip_frames.setdefault(global_frame_idx, empty_prediction())
                prediction["bbox_ls"].append(box)
                prediction["labels"].append(track_label)
                prediction["scores"].append(track_score)

    # Open the source videos directly. Decord uses the same frame indexing as
    # TGRP, so len(VideoReader) gives the complete set of global frame indices
    # that must appear in the output, including frames without a prediction.
    for clip_idx, movie_title in clip_to_movie.items():
        video_reader = VideoReader(video_paths[clip_idx], ctx=cpu(0))
        total_frames = len(video_reader)
        clip_frames = results[movie_title][clip_idx]

        invalid_indices = [
            frame_idx
            for frame_idx in clip_frames
            if not 0 <= int(frame_idx) < total_frames
        ]
        if invalid_indices:
            raise ValueError(
                f"clip {clip_idx!r} has prediction frame indices outside "
                f"[0, {total_frames}): {sorted(invalid_indices, key=int)[:10]}"
            )

        results[movie_title][clip_idx] = {
            str(frame_idx): clip_frames.get(str(frame_idx), empty_prediction())
            for frame_idx in range(total_frames)
        }

    save_dir = os.path.dirname(os.path.abspath(args.save_file))
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_file, "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        required=True,
        type=str,
        help="Root directory containing source videos (searched recursively)",
    )
    parser.add_argument("--track_file", required=True, type=str)
    parser.add_argument("--save_file", required=True, type=str)
    args = parser.parse_args()

    main()
