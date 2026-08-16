import argparse
import cv2
import copy
import gc
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
    if not scene_list:
        # A clip without a single detected cut yields no scenes at all; it is
        # one shot spanning the whole video rather than nothing to track.
        return [(0, len(VideoReader(video_file_path)))]
    return [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]

def split_video_into_shots_decord(video_file_path, shot_boundaries, frame_dir, shot_ids=None):
    """
    Split a video into temporary folders of frames based on shot boundaries using decord.

    Args:
        video_file_path (str): Path to the input video file.
        shot_boundaries (list): List of tuples (start_frame_idx, end_frame_idx) from shot_boundary_detection.
                                Note: end_frame_idx is exclusive.
        frame_dir (str): Scratch directory to write the frames into. It holds one
                         video's frames only, so shots are keyed by shot id alone.
        shot_ids (set): Shot ids to extract, or None for all of them. A resumed
                        run only needs the shots it has not written yet.
    Returns:
        list: A list of (shot_id, shot_dir) pairs, one per shot that holds at
              least one frame. Shot ids stay tied to the position in
              shot_boundaries, so they do not shift when a shot is dropped.
    """
    # Load video using decord
    vr = VideoReader(video_file_path)
    total_frames = len(vr)

    shots = []

    print(f"Splitting video into {len(shot_boundaries)} shots using decord...")

    # Process each shot
    for shot_id, (start, end) in enumerate(shot_boundaries):
        if shot_ids is not None and shot_id not in shot_ids:
            continue

        # Clip to what decord can actually decode: scenedetect reports a
        # zero-length trailing scene on some clips, and its frame count can run
        # past decord's. Either way the shot holds no frames, and an empty shot
        # directory has nothing to detect or track.
        start = max(0, min(start, total_frames))
        end = min(end, total_frames)
        if start >= end:
            print(f"Skipping empty shot {shot_id}: frame range [{start}, {end}) is empty")
            continue

        shot_dir = os.path.join(frame_dir, f"{shot_id}")
        os.makedirs(shot_dir, exist_ok=True)
        shots.append((shot_id, shot_dir))

        # Fetch frames in batch
        frame_indices = list(range(start, end))
        frames = vr.get_batch(frame_indices).asnumpy()  # Shape: (num_frames, H, W, 3)

        for idx, frame in tqdm(zip(frame_indices, frames), total=len(frame_indices), desc=f"Shot {shot_id}"):
            frame_path = os.path.join(shot_dir, f"{idx}.jpg")
            cv2.imwrite(frame_path, frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV

    return shots

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
    
def load_completed_shots(save_file):
    """
    Read back the shots a previous run already wrote to save_file.

    The JSONL is the source of truth for what is done: one line per finished
    shot, flushed as it is written. A run that is killed mid-write (SLURM
    requeues these jobs) can leave a truncated last line behind, so unparseable
    lines are dropped and the file is rewritten before anything is appended to
    it.

    Returns:
        dict: video_idx -> set of shot ids already present in the file.
    """
    completed = {}
    if not os.path.exists(save_file):
        return completed

    kept_lines = []
    dropped = 0
    with open(save_file) as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                video_idx = record["video_idx"]
                shot_idx = int(record["shot_idx"])
            except (ValueError, TypeError, KeyError):
                dropped += 1
                continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")
            completed.setdefault(video_idx, set()).add(shot_idx)

    if dropped:
        print(f"Dropping {dropped} unparseable line(s) from {save_file}")
        tmp_path = save_file + ".repair"
        with open(tmp_path, "w") as outfile:
            outfile.writelines(kept_lines)
        os.replace(tmp_path, save_file)

    return completed

def load_completed_videos(progress_file):
    """
    Read the videos a previous run finished entirely.

    This sidecar is an optimisation only: it lets a finished video be skipped
    without paying for shot detection again. Anything it misses is still caught
    by the per-shot check against the JSONL.
    """
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file) as infile:
        return {line.strip() for line in infile if line.strip()}

def mark_video_complete(progress_file, video_idx):
    with open(progress_file, "a") as outfile:
        outfile.write(video_idx + "\n")
        outfile.flush()

def release_inference_state(video_predictor, inference_state):
    """
    Drop a SAM2 inference state and hand its GPU blocks back to the allocator.

    Run on the way out of every shot, whether it finished or ran out of memory:
    the state holds that shot's image features and per-frame memory, which would
    otherwise stay resident while the next shot's state is being built.
    """
    if inference_state is not None:
        try:
            video_predictor.reset_state(inference_state)
        except Exception:
            # init_state can leave a half-built state behind when it is the call
            # that ran out of memory. There is nothing to reset in that case --
            # dropping the reference is enough.
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_file = args.save_file
    save_dir = os.path.dirname(os.path.abspath(save_file))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    progress_file = save_file + ".done"

    if args.restart:
        for path in (save_file, progress_file):
            if os.path.exists(path):
                os.remove(path)

    # Resume where the last run stopped: shots already in the output file are
    # never recomputed, and their frames are never decoded.
    completed_shots = load_completed_shots(save_file)
    completed_videos = load_completed_videos(progress_file)
    if completed_shots or completed_videos:
        done_shots = sum(len(shots) for shots in completed_shots.values())
        print(f"Resuming {save_file}: {done_shots} shot(s) done, "
              f"{len(completed_videos)} video(s) fully done")

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

    # Append, never truncate: the file carries whatever earlier runs produced.
    results = open(save_file, "a")

    # Shots skipped on CUDA OOM this run, per video, for the closing summary.
    oom_videos = {}

    # Sorted so that a resumed run walks the shard in the same order as the run
    # it is picking up from.
    for video_file in sorted(os.listdir(args.source_dir)):
        video_idx = video_file.split(".")[0]
        if video_idx in completed_videos:
            print(f"Skipping video (already finished): {video_file}")
            continue

        video_file_path = os.path.join(args.source_dir, video_file)
        print(f"Processing video: {video_file_path}")

        # Detect shot boundaries
        shot_boundaries = shot_boundary_detection(video_file_path)

        # Shot ids are positions in this list, so they are stable across runs
        # and can be matched against what is already in the output file.
        done_shot_ids = completed_shots.get(video_idx, set())
        pending_shot_ids = {i for i in range(len(shot_boundaries)) if i not in done_shot_ids}
        if not pending_shot_ids:
            print(f"Skipping video (all {len(shot_boundaries)} shots already written): {video_file}")
            mark_video_complete(progress_file, video_idx)
            continue
        if done_shot_ids:
            print(f"Resuming {video_file}: {len(done_shot_ids)} of {len(shot_boundaries)} shots already written")

        oom_shot_ids = []

        # The decoded frames are only needed while this video is being tracked,
        # so they go to a temporary directory that is removed as soon as the
        # video is done -- including when tracking raises.
        with tempfile.TemporaryDirectory(prefix="tgrp_frames_") as frame_dir:
            # Split video into shots and save frames
            shots = split_video_into_shots_decord(
                video_file_path, shot_boundaries, frame_dir, pending_shot_ids
            )

            for shot_id, shot_dir in shots:
                # A shot that exhausts the GPU takes only itself down: the
                # allocation depends on how long the shot is and how many
                # objects are being propagated, so the next shot usually fits.
                # Anything already written stays written.
                inference_state = None
                try:
                    save_shot_results = {}
                    save_shot_results["video_idx"] = video_idx
                    save_shot_results["shot_idx"] = shot_id
                    print(f"Processing frames in shot directory: {shot_dir}")

                    frame_names = [
                        p for p in os.listdir(shot_dir)
                        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                        ]
                    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                    frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]

                    if not frame_indices:
                        print(f"Skipping shot {shot_id}: no frames were written to {shot_dir}")
                        continue

                    save_shot_results["start_idx"] = min(frame_indices)
                    save_shot_results["end_idx"] = max(frame_indices)

                    # Randomly selelct 3 frames from the shot as seed frames. The
                    # draw is keyed to the shot so that re-running one -- after a
                    # requeue, say -- picks the same seed frames as before.
                    rng = random.Random(f"{video_idx}:{shot_id}")
                    n = len(frame_names)
                    if n < 3:
                        key_frame_idx = rng.sample(range(n), min(n, 3))
                    else:
                        division_size = n // 3
                        key_frame_idx = [
                            rng.choice(range(i * division_size, (i + 1) * division_size))
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
                        detections = processor.post_process_object_detection(
                            outputs=outputs, threshold=0.1, target_sizes=target_sizes
                        )
                        input_boxes = detections[0]["boxes"].cpu().numpy()
                        if input_boxes.shape[0] == 0:
                            continue
                        bbox_scores = detections[0]["scores"].cpu().numpy()

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
                except torch.cuda.OutOfMemoryError as exc:
                    # Nothing is written for this shot, so it stays absent from
                    # the output file and a later run picks it up again.
                    oom_shot_ids.append(shot_id)
                    print(f"Skipping shot {shot_id} of {video_file}: CUDA out of memory")
                    print(f"  {exc}")
                finally:
                    release_inference_state(video_predictor, inference_state)
                    inference_state = None

        if oom_shot_ids:
            # Deliberately left unmarked: the video is not finished, so a later
            # run retries exactly the skipped shots and leaves the rest alone.
            # Marking it here would drop those shots for good.
            oom_videos[video_idx] = sorted(oom_shot_ids)
            print(f"{video_file}: {len(oom_shot_ids)} shot(s) skipped on CUDA OOM, "
                  f"video left unfinished for a later run")
            continue

        # Every shot of this video is on disk, so a later run can skip it
        # without running shot detection again.
        mark_video_complete(progress_file, video_idx)

    results.close()

    if oom_videos:
        skipped = sum(len(shot_ids) for shot_ids in oom_videos.values())
        print(f"\nCUDA OOM skipped {skipped} shot(s) across {len(oom_videos)} video(s); "
              f"re-run this shard to retry them:")
        for video_idx in sorted(oom_videos):
            print(f"  {video_idx}: shots {oom_videos[video_idx]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam2_checkpoint', required=True, type=str, help='Path to SAM2 checkpoint')
    parser.add_argument('--source_dir', required=True, type=str, help='Path to source video directory')
    parser.add_argument('--save_file', required=True, type=str,
                        help='Output JSONL, one line per shot. An existing file is resumed, not overwritten')
    parser.add_argument('--restart', action='store_true',
                        help='Discard an existing output file and start the shard from scratch')
    args = parser.parse_args()

    main()
