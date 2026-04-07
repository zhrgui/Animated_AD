import argparse
import json
import os
import shutil
import copy
import subprocess
import torch
from torch.utils.data import DataLoader
from glob import glob
from PIL import Image
from tqdm import tqdm

from lwtnet import *
from dataset import DemoDataset
from trainer import DemoEvalTrainerScoreOnly
from utils import load_checkpoint


def split_list(lst, n):
    """
    Split a list into n (roughly) equal-sized chunks.
    """
    if n <= 0:
        raise ValueError("Number of chunks must be positive")
    if not lst:
        return [[] for _ in range(n)]

    q, r = divmod(len(lst), n)
    chunks = []
    start = 0

    for i in range(n):
        end = start + q + (1 if i < r else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def get_chunk(lst, n, k):
    """
    Get the k-th chunk from n chunks.
    """
    chunks = split_list(lst, n)
    if k >= len(chunks):
        raise IndexError(f"Chunk index {k} is out of range for {len(chunks)} chunks.")
    return chunks[k]


def crop_image(image, bbox, output_filename=None):
    """
    Crop an image to the region defined by the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    if output_filename is not None:
        cropped_image.save(output_filename)
    return cropped_image


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


def preprocess_tracks(tracks):
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
        frame_dir = args.frame_dir
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


def clean_folder(folder):
    """
    Delete all files and subdirectories inside a given folder.
    """
    if os.path.exists(folder) and os.listdir(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f'Failed to delete {item_path}. Reason: {e}')


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

    clip_idx_to_tracks = preprocess_tracks(all_tracks)
    vid_path_list = glob(os.path.join(args.vid_dir, "*", "*"))

    movie_title_clip_idx_list = []
    for movie_title, per_movie_predictions in audio_predictions.items():
        for clip_idx in per_movie_predictions:
            movie_title_clip_idx_list.append((movie_title, clip_idx))

    # Iterate through each clip for visually enhanced audio recognition
    for movie_title_clip_idx in tqdm(get_chunk(movie_title_clip_idx_list, args.num_chunks, args.chunk_idx), total=len(get_chunk(movie_title_clip_idx_list, args.num_chunks, args.chunk_idx))):

        movie_title, clip_idx = movie_title_clip_idx

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

        visual_clip_predictions = []
        for clip_pred in tqdm(clip_predictions, desc=f"PROCESSING {movie_title}/{clip_idx}", total=len(clip_predictions)):
            try:
                start_time = clip_pred["temporal_range"][0]
                end_time = clip_pred["temporal_range"][1]

                start_frame_idx = int(start_time * vid_fps)
                end_frame_idx = int(end_time * vid_fps)

                # Locate all overlapping tracks
                overlapped_tracks = {}
                for track in clip_tracks:
                    start_idx = track["start_idx"]
                    track_keys = list(track["track"].keys())
                    track_start = int(track_keys[0]) + start_idx
                    track_end = int(track_keys[-1]) + start_idx

                    if start_frame_idx <= track_end and end_frame_idx >= track_start:
                        # Only consider tracks with high confidence scores
                        if track["score"] <= 0.5:
                            continue

                        final_track = {}
                        for frame_idx, box in track["track"].items():
                            if int(frame_idx) >= (start_frame_idx - start_idx) and int(frame_idx) <= (end_frame_idx - start_idx):
                                final_track[frame_idx] = box

                        final_track = {k: v for k, v in sorted(final_track.items(), key=lambda item: int(item[0]))}

                        # Ignore tracks with short durations
                        if len(final_track) <= 15:
                            continue

                        track_ = copy.deepcopy(track)
                        del track_["track"]
                        track_["track"] = final_track

                        label = track_["label"]
                        if label not in overlapped_tracks or track_["score"] > overlapped_tracks[label]["score"]:
                            overlapped_tracks[label] = track_

                overlapped_tracks = list(overlapped_tracks.values())

                potential_speakers = {}

                for track in tqdm(overlapped_tracks, total=len(overlapped_tracks)):
                    shot_idx = track["shot_idx"]

                    video_path = os.path.join(args.shot_vid_dir, movie_title, year, clip_idx, f"{shot_idx:04d}.mp4")
                    audio_path = video_path.replace(args.shot_vid_dir, args.shot_aud_dir).replace("mp4", "wav")

                    # Extract audio from the video to a specified MP4 file
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    command = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ac', '1',
                        '-ar', '16000',
                        '-y',
                        audio_path
                    ]
                    try:
                        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except subprocess.CalledProcessError as e:
                        print(f"Skipping {video_path} due to error: {e}")

                    frame_dir = args.frame_dir
                    output_dir = args.output_dir
                    label = track["label"]

                    # Crop the video segment corresponding to a track.
                    # This involves extracting the relevant frames from the shot, resizing them
                    # consistently across the track, aligning with the corresponding audio segment,
                    # and saving the result as a cropped video.
                    shot_dir = os.path.join(frame_dir, movie_title, year, "shots", clip_idx, f"shot_{shot_idx}")
                    temp_vid_dir = os.path.join("tmp", "vid", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    clean_folder(temp_vid_dir)
                    temp_aud_dir = os.path.join("tmp", "aud", movie_title, year, label, clip_idx, f"shot_{shot_idx}")
                    clean_folder(temp_aud_dir)
                    output_dir_ = os.path.join(output_dir, movie_title, label)

                    output_path = crop_tracks(track["track"], video_path, audio_path, shot_dir, temp_vid_dir, temp_aud_dir, output_dir_, clip_idx, shot_idx)

                    # Use the audio-visual synchronisation model to determine whether the character in the track is speaking
                    dataset = DemoDataset(video_path=output_path, resize=args.resize, fps=args.fps, sample_rate=args.sample_rate)
                    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

                    with torch.no_grad():
                        try:
                            scores = trainer.eval(dataloader)
                            average_score = torch.mean(scores)
                            potential_speakers[label] = {"synchronization_score": average_score.item(), "classification_score": track["score"],
                                                        "track_id": track["track_id"], "set_idx": track["set_idx"], "shot_idx": track["shot_idx"]} 
                        except:
                            continue
            except:
                clip_pred["visual_prediction"] = {}
                clip_save.write(json.dumps(clip_pred) + "\n")
                continue

            clip_pred["visual_prediction"] = potential_speakers
            visual_clip_predictions.append(clip_pred)

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
    parser.add_argument( '--n_negative_samples', type=int, default=0, help='Shift range used for synchronization. E.g. set to 30 from -15 to +15 frame shifts')
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
