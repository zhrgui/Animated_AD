import argparse
import os
import cv2
import random
import shutil
import imageio
import subprocess
import concurrent.futures

from glob import glob
from tqdm import tqdm
from decord import VideoReader, cpu


def create_side_by_side_video_with_wav(
    talking_video_path: str,
    silent_video_path: str,
    output_video_path: str,
    audio_path: str
):
    talking_vr = VideoReader(talking_video_path, ctx=cpu(0))
    total_frames = len(talking_vr)
    fps = talking_vr.get_avg_fps()

    silent_vr = VideoReader(silent_video_path, ctx=cpu(0))
    silent_total_frames = len(silent_vr)
    random_silent_idx = random.randint(0, silent_total_frames - 1)
    silent_frame = silent_vr[random_silent_idx].asnumpy()

    talking_on_left = bool(random.getrandbits(1))

    silent_video_filename = f"tmp_mult/side_by_side_silent_{os.path.basename(talking_video_path)}"
    writer = imageio.get_writer(silent_video_filename, fps=fps, codec='libx264')

    for frame_id in range(total_frames):
        talking_frame = talking_vr[frame_id].asnumpy()

        talking_frame_bgr = cv2.cvtColor(talking_frame, cv2.COLOR_RGB2BGR)
        silent_frame_bgr = cv2.cvtColor(silent_frame, cv2.COLOR_RGB2BGR)

        talking_h, talking_w, _ = talking_frame_bgr.shape
        silent_h, silent_w, _ = silent_frame_bgr.shape

        if silent_h != talking_h:
            aspect_ratio = silent_w / silent_h
            new_silent_w = int(talking_h * aspect_ratio)
            silent_frame_bgr = cv2.resize(
                silent_frame_bgr,
                (new_silent_w, talking_h),
                interpolation=cv2.INTER_AREA
            )

        if talking_on_left:
            combined_bgr = cv2.hconcat([talking_frame_bgr, silent_frame_bgr])
        else:
            combined_bgr = cv2.hconcat([silent_frame_bgr, talking_frame_bgr])

        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_rgb)
    writer.close()

    merge_command = [
        "ffmpeg",
        "-y",
        "-i", silent_video_filename,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        output_video_path
    ]
    subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(silent_video_filename)


def process_videos_in_parallel(video_list: list, output_dir: str, max_workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    
    def process_video(video_path, silent_video_path):
        audio_path = video_path.replace("vid", "aud").replace("mp4", "wav")

        video_index = video_path.split("/")[-2]
        video_filename = video_path.split("/")[-1]

        output_video_path = os.path.join(output_dir, video_index, video_filename)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        output_audio_path = output_video_path.replace("vid", "aud").replace("mp4", "wav")
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        shutil.copy(audio_path, output_audio_path)

        create_side_by_side_video_with_wav(video_path, silent_video_path, output_video_path, audio_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(video_list)) as pbar:
            futures = {}
            for video_path in video_list:
                silent_video_path = random.choice([v for v in video_list if v != video_path])
                
                future = executor.submit(process_video, video_path, silent_video_path)
                futures[future] = video_path

            for future in concurrent.futures.as_completed(futures):
                talking_video_submitted = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"{talking_video_submitted} generated an exception: {exc}")
                finally:
                    pbar.update(1)


def main():
    os.makedirs("./tmp_mult", exist_ok=True)

    filelist = glob(args.video_dir)
    input_videos = [f for f in filelist if f.endswith('.mp4')]

    exist_output_filelist = glob(args.save_dir + "/*/*")

    processed_input_files = []
    unprocessed_input_files = []

    for filename in exist_output_filelist:
        video_index = filename.split("/")[-2]
        video_filename = filename.split("/")[-1]
        input_video_path = os.path.join(args.video_dir.replace("/*/*", ""), video_index, video_filename)
        processed_input_files.append(input_video_path)

    for video_path in input_videos:
        if video_path not in processed_input_files:
            unprocessed_input_files.append(video_path)

    output_directory = args.save_dir
    process_videos_in_parallel(unprocessed_input_files, output_directory, max_workers=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default="/datasets/lrs3/mp4/pretrain/*/*", type=str)
    parser.add_argument('--save_dir', default="/scratch/shared/beegfs/zhongrui/lrs3/multiple_heads/mp4", type=str)
    args = parser.parse_args()

    main()
