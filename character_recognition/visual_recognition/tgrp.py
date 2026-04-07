import argparse
import cv2
import copy
import os
import tempfile
import numpy as np
import json
import torch
import random
import networkx as nx

from PIL import Image
from tqdm import tqdm
from itertools import combinations
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector
from scipy.optimize import linear_sum_assignment

from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])

def shot_boundary_detection(video_file_path):
    """
    Detect the boundaries of all shots of a video file and return a list of tuples of frame indices.
    The tuples are of the form (start_frame_idx, end_frame_idx), in which the end_frame_idx is exclusive for the detected shot.
    """
    scene_list = detect(video_file_path, ContentDetector())
    return [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]

def split_video_into_shots_decord(video_file_path, shot_boundaries, save_frame_dir):
    """
    Split a video into temporary folders of frames based on shot boundaries using decord.

    Args:
        video_file_path (str): Path to the input video file.
        shot_boundaries (list): List of tuples (start_frame_idx, end_frame_idx) from shot_boundary_detection.
                                Note: end_frame_idx is exclusive.
    Returns:
        list: A list of paths to directories containing frames for each shot.
    """
    # Load video using decord
    vr = VideoReader(video_file_path)
    total_frames = len(vr)

    shot_dirs = []

    print(f"Splitting video into {len(shot_boundaries)} shots using decord...")

    # Process each shot
    for shot_id, (start, end) in enumerate(shot_boundaries):
        # Clip end to avoid overflow
        end = min(end, total_frames)
        video_idx = video_file_path.split("/")[-1].split(".")[0]
        shot_dir = os.path.join(save_frame_dir, video_idx, f"{shot_id}")
        os.makedirs(shot_dir, exist_ok=True)
        shot_dirs.append(shot_dir)

        # Fetch frames in batch
        frame_indices = list(range(start, end))
        frames = vr.get_batch(frame_indices).asnumpy()  # Shape: (num_frames, H, W, 3)

        for idx, frame in tqdm(zip(frame_indices, frames), total=len(frame_indices), desc=f"Shot {shot_id}"):
            frame_path = os.path.join(shot_dir, f"{idx}.jpg")
            cv2.imwrite(frame_path, frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV

    return shot_dirs

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou
    
def non_max_suppression(bboxes, scores, iou_threshold=0.5):
    """Perform non-maximum suppression to remove overlapping boxes."""
    # Sort boxes by score in descending order
    indices = np.argsort(scores)[::-1]

    selected_bboxes = []
    selected_scores = []

    while len(indices) > 0:
        # Pick the highest-score box
        current = indices[0]
        current_bbox = bboxes[current]
        current_score = scores[current]

        selected_bboxes.append(current_bbox)
        selected_scores.append(current_score)

        # Remove the selected index
        indices = indices[1:]

        if len(indices) == 0:
            break

        # Filter out boxes with high IoU overlap
        remaining_bboxes = bboxes[indices]
        ious = np.array([compute_iou(current_bbox, bbox) for bbox in remaining_bboxes])
        indices = indices[ious < iou_threshold]

    return selected_bboxes, selected_scores

def get_bounding_box_from_mask(mask):
    """Get bounding box [x_min, y_min, x_max, y_max] from a binary mask."""
    mask_2d = np.squeeze(mask)  # Remove extra dimensions if any
    non_zero_coords = np.where(mask_2d > 0)  # Get coordinates of mask pixels

    if len(non_zero_coords[0]) == 0:  # Return None if mask is empty
        return None

    # Compute bounding box corners
    y_min = int(np.min(non_zero_coords[0]))
    y_max = int(np.max(non_zero_coords[0]))
    x_min = int(np.min(non_zero_coords[1]))
    x_max = int(np.max(non_zero_coords[1]))

    return [x_min, y_min, x_max, y_max]

def merge_dicts(dict1, dict2):
    """Merge two dicts without overwriting existing keys, then sort by integer key."""
    merged_dict = copy.deepcopy(dict1)  # Start with a deep copy of dict1

    for key, value in dict2.items():
        if key not in merged_dict:      # Only add new keys from dict2
            merged_dict[key] = value

    # Sort by key (converted to int) to ensure numeric order
    merged_dict = dict(sorted(merged_dict.items(), key=lambda item: int(item[0])))

    return merged_dict

def compute_average_iou(track1, track2):
    """Compute the average IoU between two tracks across overlapping frames."""
    # Find frames that exist in both tracks
    overlapping_frames = set(track1.keys()) & set(track2.keys())
    if not overlapping_frames:
        return 0.0

    ious = []
    for frame in overlapping_frames:
        box1 = track1[frame]
        box2 = track2[frame]
        iou = compute_iou(box1, box2)  # IoU for this frame
        ious.append(iou)

    if not ious:
        return 0.0

    return sum(ious) / len(ious)

def build_cost_matrix(tracks_set_1, tracks_set_2):
    """Build a cost matrix for matching tracks based on (1 - average IoU)."""
    track_ids_1 = list(tracks_set_1.keys())
    track_ids_2 = list(tracks_set_2.keys())

    # Initialize cost matrix
    cost_matrix = np.zeros((len(track_ids_1), len(track_ids_2)), dtype=np.float32)

    for i, track_id_1 in enumerate(track_ids_1):
        for j, track_id_2 in enumerate(track_ids_2):
            track1 = tracks_set_1[track_id_1]
            track2 = tracks_set_2[track_id_2]

            # Compute cost as 1 - average IoU
            avg_iou = compute_average_iou(track1, track2)
            cost_matrix[i, j] = 1.0 - avg_iou

    return cost_matrix, track_ids_1, track_ids_2

def match_tracks(tracks_set_1, tracks_set_2, idx1, idx2, iou_threshold=0.5):
    """Match tracks between two sets using Hungarian algorithm and IoU threshold."""
    # Build cost matrix (1 - IoU) for assignment
    cost_matrix, track_ids_1, track_ids_2 = build_cost_matrix(tracks_set_1, tracks_set_2)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for row, col in zip(row_ind, col_ind):
        avg_iou = 1.0 - cost_matrix[row, col]  # Convert cost back to IoU
        if avg_iou >= iou_threshold:
            global_track_id_1 = f"set{idx1}_{track_ids_1[row]}"
            global_track_id_2 = f"set{idx2}_{track_ids_2[col]}"
            matches.append((global_track_id_1, global_track_id_2, avg_iou))

    return matches

def compute_all_pairwise_matches(rare_track_sets, iou_threshold=0.5):
    """Compute track matches between all pairs of track sets."""
    track_set_indices = range(len(rare_track_sets))
    matches = []

    # Compare every pair of track sets (combinations of two)
    for idx1, idx2 in combinations(track_set_indices, 2):
        tracks_set_1 = rare_track_sets[idx1]
        tracks_set_2 = rare_track_sets[idx2]

        # Match tracks between the two sets
        matches_pair = match_tracks(tracks_set_1, tracks_set_2, idx1, idx2, iou_threshold)
        matches.extend(matches_pair)

    return matches

def build_track_graph(matches):
    """Build a graph where tracks are nodes and matches form edges."""
    G = nx.Graph()

    for track_id_1, track_id_2, avg_iou in matches:
        G.add_node(track_id_1)  # Ensure track 1 is added
        G.add_node(track_id_2)  # Ensure track 2 is added
        G.add_edge(track_id_1, track_id_2)  # Connect matching tracks

    return G

def get_connected_components(G):
    """Return connected components from the track graph."""
    return list(nx.connected_components(G))

def select_tracks_to_keep(connected_components, global_tracks):
    """
    Select tracks to keep from connected components.
    Keep only components that span at least two different track sets.
    """
    tracks_to_keep = []

    for component in connected_components:
        sets_in_component = set()
        tracks_in_component = []

        # Collect all tracks and their originating set indices
        for global_track_id in component:
            track_info = global_tracks[global_track_id]
            set_idx = track_info['set_idx']
            sets_in_component.add(set_idx)
            tracks_in_component.append((set_idx, track_info['track_id']))

        # Keep only if component has tracks from at least two sets
        if len(sets_in_component) >= 2:
            tracks_to_keep.append(tracks_in_component)

    return tracks_to_keep

def select_representative_tracks(tracks_to_keep, rare_track_sets):
    """
    Select a representative track from each group based on the highest average IoU
    with all other tracks in the same group.
    """
    final_tracks = []

    for group in tracks_to_keep:
        avg_iou_scores = {}

        # Compute average IoU for each track against others in the group
        for set_idx_1, track_id_1 in group:
            track1 = rare_track_sets[set_idx_1][track_id_1]
            iou_sum = 0.0
            count = 0

            for set_idx_2, track_id_2 in group:
                if (set_idx_1, track_id_1) != (set_idx_2, track_id_2):
                    track2 = rare_track_sets[set_idx_2][track_id_2]
                    avg_iou = compute_average_iou(track1, track2)
                    iou_sum += avg_iou
                    count += 1

            avg_iou_scores[(set_idx_1, track_id_1)] = iou_sum / count if count > 0 else 0.0

        # Pick the track with the highest average IoU
        best_track = max(avg_iou_scores.items(), key=lambda x: x[1])[0]
        set_idx, track_id = best_track
        final_track = rare_track_sets[set_idx][track_id]

        final_tracks.append({'track_id': track_id, 'set_idx': set_idx, 'track': final_track})

    return final_tracks

def merge_tracks(track_results_per_shot):
    rare_track_sets = []

    # Extract every object tracks from the frame-level results based on the object idx
    frame_indices = sorted(track_results_per_shot.keys())
    for frame_idx in frame_indices:
        frame_data = track_results_per_shot[frame_idx]
        track_set = {}
        # Iterate through each frame
        for f_idx, item in frame_data.items():
            # Extract each object track
            for track_id, bbox in item.items():
                if track_id not in track_set:
                    track_set[track_id] = {f_idx: bbox}
                else:
                    track_set[track_id][f_idx] = bbox
        # Merge all object tracks from the seed frames
        rare_track_sets.append(track_set)

    # Unbatch the groups and list all tracks
    global_tracks = {}
    for idx, track_set in enumerate(rare_track_sets):
        for track_id in track_set.keys():
            global_track_id = f"set{idx}_{track_id}"
            global_tracks[global_track_id] = {
                'set_idx': idx,
                'track_id': track_id,
                'track': track_set[track_id]
            }

    # Find all matches between tracks across every pair from the three seed frames
    matches = compute_all_pairwise_matches(rare_track_sets, iou_threshold=0.5)

    # Build a graph based on found matches
    G = build_track_graph(matches)

    # Find the connected components
    connected_components = get_connected_components(G)

    # Keep the tracks that are at least detected from two seed frames
    tracks_to_keep = select_tracks_to_keep(connected_components, global_tracks)

    # Resolve overlapped tracks
    final_tracks = select_representative_tracks(tracks_to_keep, rare_track_sets)

    final_tracks_serializable_per_shot = []
    for track_info in final_tracks:
        track_serializable = {
            'track_id': track_info['track_id'],
            'set_idx': track_info['set_idx'],
            'track': {str(frame_idx): bbox for frame_idx, bbox in track_info['track'].items()}
        }
        final_tracks_serializable_per_shot.append(track_serializable)

    return final_tracks_serializable_per_shot
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Intializing SAM2")
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # predictor = SAM2ImagePredictor(sam2_model)
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device)

    print("Initializing Owlv2")
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(device)

    texts = [["a photo of an animated character"]]

    results = open(args.save_file, "w")

    for video_file in os.listdir(args.source_dir):
        video_file_path = os.path.join(args.source_dir, video_file)
        print(f"Processing video: {video_file_path}")

        # Detect shot boundaries
        shot_boundaries = shot_boundary_detection(video_file_path)

        # Split video into shots and save frames
        shot_dirs = split_video_into_shots_decord(video_file_path, shot_boundaries, args.save_frame_dir)

        for shot_dir in shot_dirs:
            save_shot_results = {}
            save_shot_results["video_idx"] = video_file.split(".")[0]
            save_shot_results["shot_idx"] = int(shot_dir.split("/")[-1])
            print(f"Processing frames in shot directory: {shot_dir}")

            frame_names = [
                p for p in os.listdir(shot_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]

            save_shot_results["start_idx"] = min(frame_indices)
            save_shot_results["end_idx"] = max(frame_indices)
            
            # Randomly selelct 3 frames from the shot as seed frames
            n = len(frame_names)
            if n < 3:
                key_frame_idx = random.sample(range(n), min(n, 3))
            else:
                division_size = n // 3
                key_frame_idx = [
                    random.choice(range(i * division_size, (i + 1) * division_size))
                    for i in range(3)
                ]

            # Detection results for the selected seed frames
            key_frame_detection_results = {}
            for frame_idx in key_frame_idx:
                img_path = os.path.join(shot_dir, frame_names[frame_idx])
                image = Image.open(img_path).convert("RGB")

                inputs = processor(text=texts, images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = owlv2_model(**inputs)

                target_sizes = torch.Tensor([image.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs=outputs, threshold=0.1, target_sizes=target_sizes
                )
                input_boxes = results[0]["boxes"].cpu().numpy()
                if input_boxes.shape[0] == 0:
                    continue
                bbox_scores = results[0]["scores"].cpu().numpy()

                # NMS for detected boxes
                selected_bboxes, selected_scores = non_max_suppression(input_boxes, bbox_scores, iou_threshold=0.5)

                key_frame_detection_results[str(frame_idx)] = {"bbox": selected_bboxes,
                                                            "bbox_scores": selected_scores}

            # Initiate SAM2 Propagation
            track_results_per_shot = {}

            inference_state = video_predictor.init_state(video_path=shot_dir)

            for frame_idx, anns in tqdm(key_frame_detection_results.items()):
                ann_frame_idx = int(frame_idx)
                boxes = anns["bbox"]

                for i, box in enumerate(boxes):
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=i,
                            box=box,
                        )

                # Propagate the boxes forward
                video_segments_forward = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    video_segments_forward[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Propagate the boxes backward
                video_segments_reverse = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
                    video_segments_reverse[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Reset for next propagation
                video_predictor.reset_state(inference_state)

                # Merge the forward and backward results
                video_segments_all = merge_dicts(video_segments_forward, video_segments_reverse)

                # Filter empty boxes
                video_segments = {}

                for idx, item in video_segments_all.items():
                    new_item = {}
                    for object_id, mask in item.items():
                        if not np.any(mask):
                            continue
                        bbox = get_bounding_box_from_mask(mask)
                        if box_area(bbox) <= 0:
                            continue
                        new_item[object_id] = bbox
                    video_segments[idx] = new_item

                track_results_per_shot[str(frame_idx)] = video_segments

            # Filter detected tracks through tripartite matching
            track_results = merge_tracks(track_results_per_shot)

            save_shot_results["tracks"] = track_results

            # Save shot results
            results.write(json.dumps(save_shot_results) + "\n")
            results.flush()
    
    results.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam2_checkpoint', required=True, type=str, help='Path to SAM2 checkpoint')
    parser.add_argument('--source_dir', required=True, type=str, help='Path to source video directory')
    parser.add_argument('--save_frame_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    args = parser.parse_args()

    main()
