import argparse
import os
import json
from glob import glob

def main():
    with open(args.track_file, 'r') as infile:
        tracks = json.load(infile)

    results = {}
    for track in tracks:
        movie_title = track["movie_title"]

        if movie_title not in results:
            results[movie_title] = {}

        year = track["year"]
        clip_idx = track["clip_idx"]

        if clip_idx not in results[movie_title]:
            results[movie_title][clip_idx] = {}

        shot_idx = track["shot_idx"]

        shot_folder = os.path.join(args.source_dir, clip_idx, f"{shot_idx}")
        frame_names = [p for p in os.listdir(shot_folder) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]
        shot_start_idx = min(frame_indices)

        for track_instance in track["tracks"]:
            track_sequence = track_instance["track"]
            track_label = track_instance["label"]
            track_score = track_instance["score"]

            for frame_idx, box in track_sequence.items():
                global_frame_idx = str(shot_start_idx + int(frame_idx))

                if global_frame_idx not in results[movie_title][clip_idx]:
                    results[movie_title][clip_idx][global_frame_idx] = {"bbox_ls": [], "labels": [], "scores": []}

                results[movie_title][clip_idx][global_frame_idx]["bbox_ls"].append(box)
                results[movie_title][clip_idx][global_frame_idx]["labels"].append(track_label)
                results[movie_title][clip_idx][global_frame_idx]["scores"].append(track_score)

    for movie_title, movie_frames in results.items():
        for clip_idx, clip_frames in movie_frames.items():
            vid_folder_list = glob(os.path.join(args.source_dir, clip_idx, "*"))

            for vid_folder in vid_folder_list:
                frame_names = [p for p in os.listdir(vid_folder) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

                frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]

                for frame_idx in frame_indices:
                    if str(frame_idx) not in results[movie_title][clip_idx]:
                        results[movie_title][clip_idx][str(frame_idx)] = {"bbox_ls": [], "labels": [], "scores": []}

    with open(args.save_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default="/scratch/shared/beegfs/zhongrui/cmd_movies", type=str)
    parser.add_argument('--track_file', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    args = parser.parse_args()

    main()
