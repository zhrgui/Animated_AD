import argparse
import json

def main():
    with open(args.annotation_file, 'r') as infile:
        annotations = json.load(infile)

    with open(args.track_predictions_file, 'r') as infile:
        track_predictions = json.load(infile)

    clip_preds_ls = []
    for clip_anns in annotations:
        clip_preds = {"movie_title": clip_anns["movie_title"], "year": clip_anns["year"], "clip_idx": clip_anns["clip_idx"], "predictions": []}

        for frame_anns in clip_anns["key_frames"]:
            clip_idx = clip_anns["clip_idx"]
            shot_idx = frame_anns["shot_idx"]

            tracking_boxes = None
            for shot_predictions in track_predictions:
                if clip_idx == shot_predictions["clip_idx"] and shot_idx == shot_predictions["shot_idx"]:
                    tracking_boxes = shot_predictions["tracking_boxes"]

            if tracking_boxes is None:
                bbox_ls = []
                pred_labels = []
                scores = []
            elif str(frame_anns["frame_idx"]) not in tracking_boxes:
                bbox_ls = []
                pred_labels = []
                scores = []
            else:
                bbox_ls = tracking_boxes[str(frame_anns["frame_idx"])]["bbox_ls"]
                pred_labels = tracking_boxes[str(frame_anns["frame_idx"])]["labels"]
                scores = tracking_boxes[str(frame_anns["frame_idx"])]["scores"]
            clip_preds["predictions"].append({"frame_idx": frame_anns["frame_idx"], "bbox_ls": bbox_ls, "labels": pred_labels, "cls_scores": scores})

        clip_preds_ls.append(clip_preds)

    with open(args.save_path, 'w') as outfile:
        json.dump(clip_preds_ls, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', default="../../resources/cmdam_boxes.json", type=str)
    parser.add_argument('--track_predictions_file', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    args = parser.parse_args()

    main()
