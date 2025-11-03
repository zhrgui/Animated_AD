import argparse
import numpy as np
import json
import math
import random

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm

OLD_LABEL_NEW_LABEL_MAP = [("Agnes", "Agnes Gru"), ("Edith", "Edith Gru"), ("Margo", "Margo Gru"), ("Gru", "Felonius Gru"), ("Kyle", "Kyle Gru"), ("Minion", "Minions")]

def load_json(json_file):
    with open(json_file, 'r') as infile:
        return json.load(infile)
    
def replace_labels(label_ls, old_label, new_label):
    for i in range(len(label_ls)):
        if label_ls[i] == old_label:
            label_ls[i] = new_label
    return label_ls

def process_files(prediction_file, annotation_file):
    predictions = load_json(prediction_file)
    annotations = load_json(annotation_file)

    prediction_array = []
    gt_array = []

    for clip_preds, clip_anns in tqdm(zip(predictions, annotations), total=len(predictions)):
        assert clip_preds["clip_idx"] == clip_anns["clip_idx"]

        shot_idx_ls = []
        pred_labels = []
        pred_scores = []
        gt_labels = []

        for frame_pred, frame_ann in zip(clip_preds["predictions"], clip_anns["key_frames"]):
            shot_idx_ls.append(frame_ann["shot_idx"])
            pred_labels.append(frame_pred["labels"])
            # pred_scores.append(frame_pred["scores"])
            pred_scores.append(frame_pred["cls_scores"])
            gt_labels.append(frame_ann["labels"])
        
        if clip_anns["movie_title"] == "Despicable Me":
            for (old_label, new_label) in OLD_LABEL_NEW_LABEL_MAP:
                gt_labels = [replace_labels(labels, old_label, new_label) for labels in gt_labels]

        all_labels = set()
        for labels in pred_labels + gt_labels:
            all_labels.update(labels)
        all_labels = sorted(list(all_labels))
        label_to_index = {label: idx for idx, label in enumerate(all_labels)}

        num_labels = len(all_labels)
        num_frames = len(pred_labels)

        pred_score_matrix = np.zeros((num_frames, num_labels))
        gt_binary_matrix = np.zeros((num_frames, num_labels))

        for idx, (pred_frame_labels, pred_frame_scores, gt_frame_labels) in enumerate(zip(pred_labels, pred_scores, gt_labels)):
            for label, score in zip(pred_frame_labels, pred_frame_scores):
                if pred_score_matrix[idx, label_to_index[label]] == 0:
                    pred_score_matrix[idx, label_to_index[label]] = score
                else:
                    if pred_score_matrix[idx, label_to_index[label]] < score:
                        pred_score_matrix[idx, label_to_index[label]] = score
            for label in gt_frame_labels:
                gt_binary_matrix[idx, label_to_index[label]] = 1
        
        shot_idx_to_frames = {}
        for i, shot_idx in enumerate(shot_idx_ls):
            if shot_idx not in shot_idx_to_frames.keys():
                shot_idx_to_frames[shot_idx] = [i]
            else:
                shot_idx_to_frames[shot_idx].append(i)

        merged_eval_segments = {}
        for shot_idx, frame_idx in shot_idx_to_frames.items():
            segment_idx = math.floor(shot_idx / args.num_shot)
            if segment_idx not in merged_eval_segments:
                merged_eval_segments[segment_idx] = frame_idx
            else:
                merged_eval_segments[segment_idx].extend(frame_idx)

        for segment_idx, frame_idx in merged_eval_segments.items():
            if len(frame_idx) >= 8:
                sampled_frame_idx = random.sample(frame_idx, 8)
                merged_eval_segments[segment_idx] = sampled_frame_idx

        clip_pred_array = []
        clip_gt_array = []
        for _, frame_idx in merged_eval_segments.items():
            clip_pred_array.extend(list(np.max(pred_score_matrix[frame_idx], axis=0)))
            clip_gt_array.extend(list(np.max(gt_binary_matrix[frame_idx], axis=0)))
        
        prediction_array.extend(clip_pred_array)
        gt_array.extend(clip_gt_array)

    precision, recall, _ = precision_recall_curve(gt_array, prediction_array)
    average_precision = average_precision_score(gt_array, prediction_array)
    print("Average Precision:", average_precision)

    roc_auc = roc_auc_score(gt_array, prediction_array)
    print("ROC AUC:", roc_auc)

    return recall, precision, average_precision

def main():
    process_files(args.prediction_file, args.annotation_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shot', default=3, type=int, help='Number of shots merged for prediction')
    parser.add_argument('--prediction_file', default="/scratch/shared/beegfs/zhongrui/autoad_data/results/0125_cmd_test_set_sam2.1/postprocessed_bboxes_predictions.json", type=str)
    parser.add_argument('--annotation_file', default="/work/zhongrui/transfer/annotations/cmd_animated_movies_new.json", type=str)
    args = parser.parse_args()

    main()
