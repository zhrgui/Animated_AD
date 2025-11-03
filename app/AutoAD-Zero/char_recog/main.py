import os
import json 
import argparse
import pandas as pd
import numpy as np

from decord import VideoReader, cpu
from utils import group_and_filter_id

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_json(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)


def main(args, movie_title_to_imdbid, anno_df, anno_df_with_face):
    track_predictions = load_json(args.track_preds)
    
    imdbid_to_movie_title = {value: key for key, value in movie_title_to_imdbid.items()}
    last_video = None

    for anno_idx, anno_row in tqdm(anno_df.iterrows(), total=len(anno_df)):

        imdbid = anno_row['imdbid']
        if imdbid not in imdbid_to_movie_title:
            print("Skipping non-animated movies")
            continue
        movie_title = imdbid_to_movie_title[imdbid]

        cmd_filename = anno_row['cmd_filename']

        year = cmd_filename.split("/")[0]
        clip_idx = cmd_filename.split("/")[1]

        predictions = track_predictions[movie_title][clip_idx]

        video_path = os.path.join(args.video_dir, cmd_filename + '.mkv')
        if video_path != last_video:
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
            local_fps = float(decord_vr.get_avg_fps()) 
            last_video = video_path
        
        # get frame range for the current annotation
        start = anno_row["scaled_start"]
        end = anno_row["scaled_end"]
        start_frame = int(local_fps * start)
        end_frame = min(int(local_fps * end), len(decord_vr) - 1)

        bboxes_all_frames = []
        pred_id_all_frames = []
        pred_cos_all_frames = []

        for i in range(start_frame, end_frame + 1):
            if str(i) not in predictions:
                bboxes_all_frames.append([])
                pred_id_all_frames.append([])
                pred_cos_all_frames.append([])
            else:
                bboxes_all_frames.append(predictions[str(i)]['bbox_ls'])
                pred_id_all_frames.append(predictions[str(i)]['labels'])
                pred_cos_all_frames.append(predictions[str(i)]['scores'])

        # sample 16 frames uniformly
        num_frames = 16
        frame_id_list_sampled = np.linspace(0, end_frame - start_frame, num_frames, endpoint=False, dtype=int)
        try:
            sampled_bboxes_all_frames = [bboxes_all_frames[i] for i in frame_id_list_sampled]
            sampled_pred_id_all_frames = [pred_id_all_frames[i] for i in frame_id_list_sampled]
            sampled_pred_cos_all_frames = [pred_cos_all_frames[i] for i in frame_id_list_sampled]
        except:
            import ipdb; ipdb.set_trace()

        # match, filter and format 
        index_filtered_dict = group_and_filter_id(sampled_pred_id_all_frames, sampled_pred_cos_all_frames, args.score_thresh)
        
        anno_row["bboxes"] = sampled_bboxes_all_frames
        anno_row["pred_ids"] = index_filtered_dict
        anno_df_with_face = pd.concat([anno_df_with_face, pd.DataFrame([anno_row])], ignore_index=True)
    return anno_df_with_face

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cmdad", type=str)
    parser.add_argument('--anno_path', default="/scratch/shared/beegfs/zhongrui/autoad_data/annotations/animated_cmd_ad_anno_v1.csv", type=str)
    parser.add_argument('--track_preds', default="/scratch/shared/beegfs/zhongrui/autoad_data/results/0404_videollama2/cmd_animated_set_frame_predictions.json", type=str)
    parser.add_argument('--video_dir', default="/scratch/shared/beegfs/zhongrui/animated_videos", type=str)
    parser.add_argument('--output_dir', default="/scratch/shared/beegfs/zhongrui/autoad_data/results/0406_visualization", type=str)

    parser.add_argument('--movie_title_to_imdbid_file', default="/scratch/shared/beegfs/zhongrui/autoad_data/annotations/movie_title_to_imdb.json", type=str)
    parser.add_argument('--score_thresh', default=0.45, type=float, help='threshold on the cosine similarity for character recognition')
    args = parser.parse_args()

    # load movie_title_to_imdb
    movie_title_to_imdbid = load_json(args.movie_title_to_imdbid_file)

    # load AD annotaion file
    anno_df = pd.read_csv(args.anno_path)

    print(f"In total {len(anno_df)} AD annotations")
    anno_df["bboxes"] = [[[]] * 16] * len(anno_df)
    anno_df["pred_ids"] = [{}] * len(anno_df)

    # create a new file to save character recognition results
    anno_df_with_char = pd.DataFrame(columns=anno_df.columns)

    # save face bboxes and ids into annotation file
    anno_df_with_face = main(args, movie_title_to_imdbid, anno_df, anno_df_with_char)

    os.makedirs(args.output_dir, exist_ok=True)
    anno_df_with_face.to_csv(os.path.join(args.output_dir, f"{args.dataset}_anno_with_face_{args.score_thresh}.csv"), index=False)
