import argparse
import json
import numpy as np

OLD_LABEL_NEW_LABEL_MAP = [
    ("Agnes", "Agnes Gru"),
    ("Edith", "Edith Gru"),
    ("Margo", "Margo Gru"),
    ("Gru", "Felonius Gru"),
    ("Kyle", "Kyle Gru"),
    ("Minion", "Minions"),
    ("Bob the Minion", "Minions"),
    ("Tim the Minion", "Minions"),
    ("Manfred", "Manny"),
    ("Other Mother", "The Beldam (Other Mother)"),
    ("Uncle Aaron", "Aaron Davis"),
    ("Jefferson Davis", "Jefferson Morales"),
    ("Hero Boy", "Hero Boy (Chris)"),
]

label_mapping = {}
for (old_label, new_label) in OLD_LABEL_NEW_LABEL_MAP:
    label_mapping[old_label] = new_label


def normalize_label(label):
    return label_mapping.get(label, label)

def load_json(file_path):
    with open(file_path) as infile:
        return json.load(infile)

def calculate_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    return area

def non_max_suppression(bboxes, scores, labels, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on bounding boxes with associated scores.
    """
    if len(bboxes) == 0:
        return [], [], []

    bboxes = np.array(bboxes)
    scores = np.array(scores)

    indices = np.argsort(scores)[::-1]

    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        current_bbox = bboxes[current_index]

        indices = indices[1:]

        if len(indices) == 0:
            break

        remaining_bboxes = bboxes[indices]

        ious = np.array([compute_iou(current_bbox, bbox) for bbox in remaining_bboxes])

        indices = indices[ious < iou_threshold]

    selected_bboxes = bboxes[selected_indices]
    selected_scores = scores[selected_indices]

    selected_labels = []
    for index in selected_indices:
        selected_labels.append(labels[index])

    return selected_bboxes.tolist(), selected_scores.tolist(), selected_labels

def convert_anns_to_eval_format(file_path, is_prediction=False):
    boxes = []

    data = load_json(file_path)
    
    for clip_anns in data:
        clip_idx = clip_anns["clip_idx"]

        if not is_prediction:
            clip_box_anns = clip_anns["key_frames"]
        else:
            clip_box_anns = clip_anns["predictions"]

        for frame_anns in clip_box_anns:
            frame_idx = frame_anns["frame_idx"]
            image_id = f"{clip_idx}_{frame_idx}"

            bbox_ls = frame_anns["bbox_ls"]
            labels = frame_anns["labels"]

            if is_prediction:
                scores = frame_anns["cls_scores"]

                bbox_ls, scores, labels = non_max_suppression(bbox_ls, scores, labels, iou_threshold=0.5)

                for bbox, label, score in zip(bbox_ls, labels, scores):
                    # Ignore small boxes
                    if calculate_bbox_area(bbox) < 36 * 36:
                        continue

                    boxes.append({"image_id": image_id,
                                  "bbox": bbox,
                                  "label": label.split("/")[0],
                                  "score": score})
            else:
                for bbox, label in zip(bbox_ls, labels):
                    if calculate_bbox_area(bbox) < 36 * 36:
                        continue

                    boxes.append({"image_id": image_id,
                                  "bbox": bbox,
                                  "label": label})
    return boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

def main():
    pred_boxes = convert_anns_to_eval_format(args.prediction_file, is_prediction=True)
    gt_boxes = convert_anns_to_eval_format(args.annotation_file, is_prediction=False)

    num_gt = len(gt_boxes)
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    ap_sum = []
    for iou_threshold in iou_thresholds:

        gt_dict = {}
        for gt_box in gt_boxes:
            image_id = gt_box['image_id']
            if image_id not in gt_dict:
                gt_dict[image_id] = []
            gt_dict[image_id].append({'bbox': gt_box['bbox'], 'label': gt_box['label'], 'detected': False})
        
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
    
        tp = []
        fp = []
        scores = []

        for pred_box in pred_boxes_sorted:
            image_id = pred_box['image_id']
            pred_bbox = pred_box['bbox']
            pred_score = pred_box['score']
            pred_label = pred_box['label']

            matched = False

            if image_id in gt_dict:
                gt_boxes_in_image = gt_dict[image_id]
                best_iou = 0
                best_gt_idx = -1

                for idx, gt_entry in enumerate(gt_boxes_in_image):
                    if not gt_entry['detected']:
                        gt_bbox = gt_entry['bbox']
                        iou = compute_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_gt_idx >= 0 and best_iou >= iou_threshold:
                    gt_label = gt_boxes_in_image[best_gt_idx]['label']
                    if normalize_label(pred_label) == normalize_label(gt_label):
                        matched = True
                        gt_boxes_in_image[best_gt_idx]['detected'] = True

            if matched:
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

            scores.append(pred_score)

        if len(tp) == 0:
            print("No predictions to evaluate.")
            continue

        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        ap_sum.append(ap)
        # print(f"Average Precision (AP): {ap} @ IoU: {iou_threshold}")

    mAP = sum(ap_sum) / len(ap_sum)
    print(f"mAP: {mAP}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', required=True, type=str, help='Path to prediction JSON file')
    parser.add_argument('--annotation_file', required=True, type=str, help='Path to annotation JSON file')
    args = parser.parse_args()

    main()
