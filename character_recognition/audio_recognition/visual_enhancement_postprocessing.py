"""
Merge the per-clip visual enhancement results into one prediction file.

A segment keeps its voice-bank prediction unless the voice bank was unsure about
it (score below --threshold) and a character track scored higher on the combined
synchronisation x classification score.
"""

import argparse
import os
import json
from glob import glob

from utils.io import load_jsonl


def main():
    predictions = {}
    for jsonl_file in glob(os.path.join(args.result_dir, "*", "*")):
        movie_title = jsonl_file.split("/")[-2]
        clip_idx = jsonl_file.split("/")[-1].split(".")[0]
        if movie_title not in predictions:
            predictions[movie_title] = {}
        predictions[movie_title][clip_idx] = load_jsonl(jsonl_file)

    new_predictions = {}
    for movie_title, per_movie_predictions in predictions.items():
        if movie_title not in new_predictions:
            new_predictions[movie_title] = {}

        for clip_idx, clip_preds in per_movie_predictions.items():
            new_clip_preds = []

            for pred in clip_preds:
                if "visual_prediction" not in pred:
                    pred["visual_prediction"] = []

                if len(pred["visual_prediction"]) == 0 or pred["score"] >= args.threshold:
                    new_clip_preds.append(pred)
                    continue

                visual_pred = pred["visual_prediction"]
                del pred["visual_prediction"]

                max_score = 0.
                max_character = None
                for character, scores in visual_pred.items():

                    score = scores["synchronization_score"] * scores["classification_score"] * args.alpha

                    if score >= max_score:
                        max_score = score
                        max_character = character

                if max_score >= pred["score"]:
                    pred["label"] = max_character
                    pred["score"] = max_score

                new_clip_preds.append(pred)

            new_predictions[movie_title][clip_idx] = new_clip_preds

    with open(args.save_file, 'w') as outfile:
        json.dump(new_predictions, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--alpha', default=None, type=float)
    args = parser.parse_args()

    main()
