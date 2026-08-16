import argparse
import json


def index_shot_predictions(track_predictions):
    """Index shot records by clip and shot, rejecting ambiguous duplicates."""
    predictions_by_shot = {}
    for prediction in track_predictions:
        clip_idx = prediction.get("clip_idx", prediction.get("video_idx"))
        if clip_idx is None:
            raise KeyError("prediction has neither 'clip_idx' nor 'video_idx'")

        key = (str(clip_idx), int(prediction["shot_idx"]))
        if key in predictions_by_shot:
            raise ValueError(
                f"duplicate predictions for clip {clip_idx!r}, "
                f"shot {prediction['shot_idx']!r}"
            )
        predictions_by_shot[key] = prediction
    return predictions_by_shot


def predictions_for_frame(shot_prediction, frame_idx):
    """Return boxes, labels, and scores for one clip-global frame index."""
    if shot_prediction is None:
        return [], [], []

    # Backward compatibility with the format select_subset.py originally read.
    if "tracking_boxes" in shot_prediction:
        frame_prediction = shot_prediction["tracking_boxes"].get(str(frame_idx))
        if frame_prediction is None:
            return [], [], []
        return (
            frame_prediction["bbox_ls"],
            frame_prediction["labels"],
            frame_prediction["scores"],
        )

    # classification.py emits one shot record containing object tracks. Track
    # frame keys are relative to the shot; annotation frame_idx is clip-global.
    if "tracks" not in shot_prediction:
        raise KeyError(
            "shot prediction has neither 'tracking_boxes' nor 'tracks'"
        )
    if "start_idx" not in shot_prediction:
        raise KeyError("track-based shot prediction has no 'start_idx'")

    local_frame_idx = str(int(frame_idx) - int(shot_prediction["start_idx"]))
    bbox_ls = []
    labels = []
    scores = []
    for track in shot_prediction["tracks"]:
        bbox = track["track"].get(local_frame_idx)
        if bbox is not None:
            bbox_ls.append(bbox)
            labels.append(track["label"])
            scores.append(track["score"])
    return bbox_ls, labels, scores


def main():
    with open(args.annotation_file, 'r') as infile:
        annotations = json.load(infile)

    with open(args.track_predictions_file, 'r') as infile:
        track_predictions = json.load(infile)
    predictions_by_shot = index_shot_predictions(track_predictions)

    clip_preds_ls = []
    for clip_anns in annotations:
        clip_preds = {"movie_title": clip_anns["movie_title"], "year": clip_anns["year"], "clip_idx": clip_anns["clip_idx"], "predictions": []}

        for frame_anns in clip_anns["key_frames"]:
            shot_prediction = predictions_by_shot.get(
                (str(clip_anns["clip_idx"]), int(frame_anns["shot_idx"]))
            )
            bbox_ls, pred_labels, scores = predictions_for_frame(
                shot_prediction, frame_anns["frame_idx"]
            )
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
