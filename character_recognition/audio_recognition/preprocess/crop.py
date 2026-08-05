"""
Turning a character track into a clip.

Both the voice-bank builder (`build_voice_examples.py`) and the visual
enhancement of the audio predictions (`../visual_enhancement.py`) need the same
thing: take the boxes of one track, cut them out of the shot's frames at a
constant crop size, and mux them back with the matching slice of audio.
"""

import os
import shutil
import subprocess

from PIL import Image
from tqdm import tqdm


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


def extract_audio(video_path, audio_path, sample_rate=16000, channels=1):
    """Extract the audio track of a video as a mono wav."""
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ac', str(channels),
        '-ar', str(sample_rate),
        '-y',
        audio_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


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
