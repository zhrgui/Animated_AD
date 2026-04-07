import argparse
import os
import subprocess
import json
import torch
import torchaudio
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from decord import VideoReader, cpu
from utils import load_json


def crop_image(image, bbox, output_filename=None):
    """
    Crop an image to the region defined by the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    if output_filename is not None:
        cropped_image.save(output_filename)
    return cropped_image


def filter_repeating_characters(tracks):
    """
    Filter out duplicate character tracks by keeping only the highest-scoring track per character,
    except for "Minions" where all tracks are retained.
    """
    char_tracks = {}
    allowed_tracks = []

    for track in tracks:
        if track["label"] == "Minions":
            allowed_tracks.append(track)
        if track["label"] not in char_tracks.keys():
            char_tracks[track["label"]] = track
        else:
            if track["score"] > char_tracks[track["label"]]["score"]:
                char_tracks[track["label"]] = track

    return list(char_tracks.values()) + allowed_tracks


def slide_box_into_frame(box, frame_width, frame_height):
    """
    Adjust a bounding box so that it stays fully within the frame dimensions.
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width > frame_width or box_height > frame_height:
        raise ValueError("Box dimensions exceed frame dimensions")

    if x1 < 0:
        x1 = 0
        x2 = box_width
    elif x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - box_width

    if y1 < 0:
        y1 = 0
        y2 = box_height
    elif y2 > frame_height:
        y2 = frame_height
        y1 = frame_height - box_height
    return [x1, y1, x2, y2]


def crop_tracks(track, video_path, audio_path, shot_dir, temp_vid_dir, temp_aud_dir, save_dir, clip_idx, shot_idx):
    """
    Crop a sequence of frames corresponding to a track, align the crop size across
    the whole track, extract the corresponding audio, and save the cropped video.
    """
    from decord import VideoReader, cpu

    # Load video metadata
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    # Get frame indices for the current shot
    frame_ls = [f for f in os.listdir(shot_dir) if f.endswith('.jpg')]
    numbers = [int(f[:-4]) for f in frame_ls]
    start_idx = min(numbers)

    # Determine maximum width and height of boxes across the track
    max_width = 0
    max_height = 0
    for box in track.values():
        max_width = max(max_width, box[2] - box[0])
        max_height = max(max_height, box[3] - box[1])

    # Absolute frame indices in the video
    track_start_idx = int(list(track.keys())[0]) + start_idx
    track_end_idx = int(list(track.keys())[-1]) + start_idx

    # Iterate over each frame in the track and crop it
    for frame_idx, box in tqdm(track.items(), total=len(track)):
        # Load original frame
        frame = Image.open(os.path.join(shot_dir, f"{start_idx + int(frame_idx)}.jpg"))
        frame_width, frame_height = frame.size

        # Save path for cropped frame
        save_frame_path = os.path.join(temp_vid_dir, f"{start_idx + int(frame_idx)}.jpg")
        os.makedirs(os.path.dirname(save_frame_path), exist_ok=True)

        # Center crop around track box but keep size consistent (max_width, max_height)
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        new_box = [
            x_center - max_width / 2,
            y_center - max_height / 2,
            x_center + max_width / 2,
            y_center + max_height / 2,
        ]

        # Ensure crop box fits inside frame
        new_box = slide_box_into_frame(new_box, frame_width, frame_height)

        # Crop and save frame
        crop_image(frame, new_box, save_frame_path)

    # Compute relative start and end indices for audio alignment
    relative_start_idx = track_start_idx - start_idx
    relative_end_idx = track_end_idx - start_idx
    start_time = 0 if relative_start_idx == 0 else (relative_start_idx + 1) / fps
    end_time = (relative_end_idx + 1) / fps

    # Extract corresponding audio segment
    cropped_audio_path = os.path.join(temp_aud_dir, "audio.wav")
    os.makedirs(os.path.dirname(cropped_audio_path), exist_ok=True)
    extract_cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start_time), "-to", str(end_time),
        cropped_audio_path,
    ]
    subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Assemble cropped frames + audio into final video
    out_video_name = os.path.join(save_dir, f"{clip_idx}_{shot_idx:04d}.mp4")
    os.makedirs(os.path.dirname(out_video_name), exist_ok=True)
    image_pattern = os.path.join(temp_vid_dir, "%d.jpg")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(track_start_idx),
        "-i", image_pattern,
        "-i", cropped_audio_path,
        "-frames:v", str(track_end_idx - track_start_idx + 1),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        out_video_name,
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_video_name


def main():
    # Load the detected tracks from visual character recognition
    track_file = load_json(args.track_results)

    # Initialize retrieved tracks for potential in-movie audio exemplars
    retrieved_character_information = defaultdict(dict)
    key_to_tracks = defaultdict(dict)

    for shot_annotations in track_file:
        movie_title = shot_annotations["movie_title"]

        if args.movie_title:
            if movie_title != args.movie_title:
                continue

        year = shot_annotations["year"]
        clip_idx = shot_annotations["clip_idx"]
        shot_idx = shot_annotations["shot_idx"]
        
        key = (movie_title, year, clip_idx, shot_idx)

        tracks = shot_annotations["classified_tracks"]
        filtered_tracks = filter_repeating_characters(tracks)

        key_to_tracks[key] = filtered_tracks

        for track in filtered_tracks:
            label = track["label"]
            score = track["score"]
            
            if key not in retrieved_character_information[label] or score > retrieved_character_information[label][key]:
                retrieved_character_information[label][key] = score

    top10_by_label = {}
    for label, shots in retrieved_character_information.items():
        sorted_shots = sorted(shots.items(), key=lambda x: x[1], reverse=True)
        top10_by_label[label] = sorted_shots

    minimum_number = 10
    instance_confidence = {}
    for label, track_instances in top10_by_label.items():
        num_examples_by_label = 0
        saved_track_instances = []
        instance_confidence[label] = {}

        for track_instance in track_instances:
            try:
                score = track_instance[1]
                
                if score <= args.thres:
                    continue

                movie_title = track_instance[0][0]
                year = track_instance[0][1]
                clip_idx = track_instance[0][2]
                shot_idx = track_instance[0][3]

                instance_track = None
                for track in key_to_tracks[track_instance[0]]:
                    if track["label"] == label:
                        instance_track = track["track"]

                video_path = os.path.join(args.video_dir, movie_title, year, clip_idx, f"{shot_idx:04d}.mp4")
                audio_path = video_path.replace("vid", "aud").replace("mp4", "wav")

                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                command = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ac', '1',
                    '-ar', '16000',
                    '-y',
                    audio_path
                ]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                shot_dir = os.path.join(args.frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")
                temp_vid_dir = os.path.join("tmp/", "vid", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                temp_aud_dir = os.path.join("tmp/", "aud", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                output_dir = os.path.join(args.output_dir, movie_title, label)
                output_path = crop_tracks(instance_track, video_path, audio_path, shot_dir, temp_vid_dir, temp_aud_dir, output_dir, clip_idx, shot_idx)

                instance_confidence[label][output_path] = score
                saved_track_instances.append(track_instance)
                num_examples_by_label += 1
            except:
                continue

        if num_examples_by_label < minimum_number:
            track_instances.sort(key=lambda x: x[1], reverse=True)

            for track_instance in track_instances:
                try:
                    if track_instance in saved_track_instances or num_examples_by_label >= minimum_number:
                        continue

                    movie_title = track_instance[0][0]
                    year = track_instance[0][1]
                    clip_idx = track_instance[0][2]
                    shot_idx = track_instance[0][3]

                    instance_track = None
                    for track in key_to_tracks[track_instance[0]]:
                        if track["label"] == label:
                            instance_track = track["track"]

                    video_path = os.path.join(args.video_dir, movie_title, year, clip_idx, f"{shot_idx:04d}.mp4")
                    audio_path = video_path.replace("vid", "aud").replace("mp4", "wav")

                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    command = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ac', '1',
                        '-ar', '16000',
                        '-y',
                        audio_path
                    ]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    shot_dir = os.path.join(args.frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")
                    temp_vid_dir = os.path.join("tmp/", "vid", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    temp_aud_dir = os.path.join("tmp/", "aud", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    output_dir = os.path.join(args.output_dir, movie_title, label)
                    output_path = crop_tracks(instance_track, video_path, audio_path, shot_dir, temp_vid_dir, temp_aud_dir, output_dir, clip_idx, shot_idx)

                    instance_confidence[label][output_path] = score
                    saved_track_instances.append(track_instance)
                    num_examples_by_label += 1
                except:
                    continue

    save_file = os.path.join(args.output_dir, "instance_confidence.json")
    with open(save_file, 'w') as outfile:
        json.dump(instance_confidence, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_results', required=True, type=str, help='Path to track results JSON')
    parser.add_argument('--video_dir', required=True, type=str, help='Directory containing shot video files')
    parser.add_argument('--frame_dir', required=True, type=str, help='Directory containing frame images')
    parser.add_argument('--output_dir', required=True, type=str, help='Output directory for instance videos')
    parser.add_argument('--movie_title', default=None, type=str)
    parser.add_argument('--thres', default=0.6, type=float)

    args = parser.parse_args()

    main()
