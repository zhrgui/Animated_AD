import argparse
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

def main(args):
    with open(args.prediction_file, 'r') as infile:
        predictions = json.load(infile)

    movie_scores = {}
    all_scores = []
    
    for movie_title, per_movie_predictions in predictions.items():
        movie_scores[movie_title] = []
        
        for clip_idx, per_clip_predictions in per_movie_predictions.items():
            for clip_pred in per_clip_predictions:
                pred_label = clip_pred["prediction"]
                gt_label = clip_pred["character"]

                # Handling Annotation Inconsistencies
                if gt_label == "Felonious Gru":
                    gt_label = "Felonius Gru"
                elif gt_label == "Nancy" or gt_label == "Nana Noodleman":
                    gt_label = "Other"
                    
                score = clip_pred["score"]

                # Open Set Recognition (OSR) Adjustment
                if args.osr:
                    threshold = 0.15
                    alpha = 0.15
                    if score <= threshold:
                        pred_label = "Other"
                        score = (-score + 1) * alpha
                
                is_correct = 1 if (pred_label == gt_label) else 0
                
                movie_scores[movie_title].append((score, is_correct))
                all_scores.append((score, is_correct))
    
    movie_APs = {}
    
    for movie_title, score_label_pairs in movie_scores.items():
        scores = np.array([x[0] for x in score_label_pairs])
        is_correct = np.array([x[1] for x in score_label_pairs])
        
        # precision, recall, thresholds = precision_recall_curve(is_correct, scores)
        ap = average_precision_score(is_correct, scores)
        movie_APs[movie_title] = ap

        # display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap)
        # display.plot()
        # plt.title(f'PR Curve: {movie_title}')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.savefig(os.path.join(args.output_dir, f'{movie_title}.png'))
        # plt.close()

    if all_scores:
        total_scores = np.array([x[0] for x in all_scores])
        total_is_correct = np.array([x[1] for x in all_scores])
        
        total_precision, total_recall, _ = precision_recall_curve(total_is_correct, total_scores)
        total_ap = average_precision_score(total_is_correct, total_scores)

        # display = PrecisionRecallDisplay(precision=total_precision, recall=total_recall, average_precision=total_ap)
        # display.plot()
        # plt.title(f'Total Precision-Recall Curve\nAP={total_ap:.4f}')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.savefig(os.path.join(args.output_dir, 'total.png'))
        # plt.close()
    else:
        total_ap = 0.0
    
    results = {}
    print("\nAverage Precision (AP) scores:")
    for movie_title, ap_value in movie_APs.items():
        print(f"  {movie_title}: {ap_value:.4f}")
        results[movie_title] = ap_value
    
    print(f"  TOTAL: {total_ap:.4f}")
    results["total"] = total_ap

    # with open(os.path.join(args.output_dir, "ap_results.json"), "w") as outfile:
    #     json.dump(results, outfile, indent=4)
    # print(f"Saved AP results to {os.path.join(args.output_dir, 'ap_results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', default="/users/zhongrui/avobjects_zgui/eval/0408/predictions/seed_0/clustering_voice_bank_only_15_visual_correction.json", type=str)
    parser.add_argument('--osr', action="store_true")
    args = parser.parse_args()
    main(args)

