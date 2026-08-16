import argparse
import copy
import gc
import os
import json
import random
import numpy as np
import torch
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, binary_erosion
from decord import VideoReader, cpu

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_MOVIE_TO_VIDEO_FILES = ""
DEFAULT_SOURCE_DIR = ""
DEFAULT_CKPT_DIR = ""
DEFAULT_CHAR_FEAT_DIR = ""
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}

def load_jsonl(filepath):
    """Load a JSONL file into a list of json objects."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json(filepath):
    """Load a JSON file from the given path."""
    with open(filepath, 'r') as f:
        return json.load(f)

def reverse_movie_to_video_files(movie_to_video_files):
    """Build a video-id-to-movie mapping from movie-to-video-file entries."""
    video_idx_to_movie_titles = {}
    for movie_title, video_files in movie_to_video_files.items():
        for video_file in video_files:
            video_idx = os.path.splitext(os.path.basename(video_file))[0]
            previous_title = video_idx_to_movie_titles.get(video_idx)
            if previous_title is not None and previous_title != movie_title:
                raise ValueError(
                    f"video {video_idx!r} belongs to both "
                    f"{previous_title!r} and {movie_title!r}"
                )
            video_idx_to_movie_titles[video_idx] = movie_title
    return video_idx_to_movie_titles

def index_videos(source_dir):
    """Index a nested source-video tree by extension-free clip id."""
    videos = {}
    for root, dirnames, filenames in os.walk(source_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            video_idx, extension = os.path.splitext(filename)
            if extension.lower() not in VIDEO_EXTENSIONS:
                continue
            video_path = os.path.join(root, filename)
            if video_idx in videos:
                raise ValueError(
                    f"duplicate video id {video_idx!r}: "
                    f"{videos[video_idx]} and {video_path}"
                )
            videos[video_idx] = video_path
    return videos

def crop_object(image, bbox, mask):
    """
    Crop an object from an image using its bounding box, optionally refining
    with a mask (erosion + hole filling) and placing it on a white background.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]

    if mask is not None:
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        white_background = np.ones_like(cropped_image) * 255

        mask_bool = cropped_mask.astype(bool)
        mask_bool = binary_erosion(mask_bool, iterations=1)
        mask_bool = binary_fill_holes(mask_bool)

        if cropped_image.ndim == 3 and cropped_image.shape[2] > 1:
            mask_bool = mask_bool[..., None]

        result_image = np.where(mask_bool, cropped_image, white_background)
    else:
        result_image = cropped_image

    result_image_uint8 = result_image.astype(np.uint8)
    result_pil = Image.fromarray(result_image_uint8)
    return result_pil

def load_character_bank(character_bank_dir):
    """Load the character bank built with profile images from a directory."""
    if not os.path.exists(character_bank_dir):
        raise FileNotFoundError(f"character bank does not exist: {character_bank_dir}")
    character_bank = {}
    for file in sorted(os.listdir(character_bank_dir)):
        if not file.lower().endswith(".npz"):
            continue
        char_name = os.path.splitext(file)[0].split("_")[0]
        feature_path = os.path.join(character_bank_dir, file)
        try:
            with np.load(feature_path) as archive:
                char_feature = archive['feature'].copy()
        except (EOFError, OSError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid character feature archive: {feature_path}") from exc
        if char_feature.ndim != 1:
            raise ValueError(
                f"expected a 1D character feature in {feature_path}, "
                f"found shape {char_feature.shape}"
            )
        character_bank.setdefault(char_name, []).append(char_feature)
    if not character_bank:
        raise ValueError(f"no .npz character features found in {character_bank_dir}")
    return {
        char_name: np.stack(features)
        for char_name, features in character_bank.items()
    }

def cosine_similarity(arr1, arr2):
    """
    Compute the cosine similarity between two arrays.
    Returns a value in [-1, 1], or 0.0 if either array is all zeros.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    dot_product = np.dot(arr1, arr2)
    magnitude_arr1 = np.linalg.norm(arr1)
    magnitude_arr2 = np.linalg.norm(arr2)

    if magnitude_arr1 == 0 or magnitude_arr2 == 0:
        return 0.0
    cosine_sim = dot_product / (magnitude_arr1 * magnitude_arr2)
    return cosine_sim

def main():
    # The available metadata is movie -> [year/video_id, ...]. Reverse it once
    # so each TGRP record can look its movie up by video_idx.
    movie_to_video_files = load_json(args.movie_to_video_files)
    video_idx_to_movie_titles_mapping = reverse_movie_to_video_files(movie_to_video_files)
    video_paths = index_videos(args.source_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = None
    if args.mask:
        if not args.sam2_checkpoint:
            raise ValueError("--sam2_checkpoint is required with --mask")
        print("Initializing SAM2")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, args.sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

    print("Initializing DINOv2")
    model_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    model_arch = model_archs[args.model_size]
    model_name = f"dinov2_{model_arch}_reg"

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load the unclassified region proposal results
    unclassified_track_results = load_jsonl(args.track_file)

    missing_mapping = sorted({
        track["video_idx"] for track in unclassified_track_results
        if track["video_idx"] not in video_idx_to_movie_titles_mapping
    })
    if missing_mapping:
        raise KeyError(
            f"{len(missing_mapping)} video id(s) are absent from "
            f"{args.movie_to_video_files}: {missing_mapping[:10]}"
        )

    missing_videos = sorted({
        track["video_idx"] for track in unclassified_track_results
        if track["video_idx"] not in video_paths
    })
    if missing_videos:
        raise FileNotFoundError(
            f"{len(missing_videos)} source video(s) are absent from "
            f"{args.source_dir}: {missing_videos[:10]}"
        )

    # The multi-GPU splitter keeps a movie on one worker. Sorting here makes
    # that worker load each movie's large DINOv2 checkpoint exactly once.
    unclassified_track_results.sort(key=lambda track: (
        video_idx_to_movie_titles_mapping[track["video_idx"]],
        track["video_idx"],
        int(track["shot_idx"]),
    ))

    last_movie_title = None
    loaded_video_idx = None
    video_reader = None
    model = None

    classified_track_results = []
    for track in tqdm(unclassified_track_results, desc="Classifying shots"):
        current_video_idx = track["video_idx"]
        current_movie_title = video_idx_to_movie_titles_mapping[current_video_idx]

        # Avoid redundant model loading
        if last_movie_title != current_movie_title:
            if model is not None:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Load test-time finetuned DINOv2 weights
            pretrained_weights = os.path.join(args.ckpt_dir, current_movie_title, "finetuned_dinov2_weights.pth")
            model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, pretrained=False)
            state_dict = torch.load(pretrained_weights, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            del state_dict
            model.eval()
            model.to(device)

            # Load the character feature bank for the movie
            character_bank = load_character_bank(os.path.join(args.char_feat_dir, current_movie_title))

            last_movie_title = current_movie_title

        if loaded_video_idx != current_video_idx:
            video_reader = VideoReader(video_paths[current_video_idx], ctx=cpu(0))
            loaded_video_idx = current_video_idx

        k = 5
        classified_tracks = []
        for unclassified_track in track["tracks"]:
            # Randomly select frames across the track for character recognition
            frame_bbox_dict = unclassified_track["track"]
            frame_idx_ls = list(frame_bbox_dict.keys())
            if k < len(frame_idx_ls):
                rng = random.Random(
                    f"{current_video_idx}:{track['shot_idx']}:"
                    f"{unclassified_track.get('track_id', '')}"
                )
                selected_frames = rng.sample(frame_idx_ls, k)
            else:
                selected_frames = frame_idx_ls

            global_frame_indices = [
                int(track["start_idx"]) + int(selected_frame_idx)
                for selected_frame_idx in selected_frames
            ]
            invalid_indices = [
                frame_idx for frame_idx in global_frame_indices
                if not 0 <= frame_idx < len(video_reader)
            ]
            if invalid_indices:
                raise IndexError(
                    f"frames {invalid_indices} are outside video "
                    f"{current_video_idx} with {len(video_reader)} frames"
                )

            decoded_frames = video_reader.get_batch(global_frame_indices).asnumpy()
            image_tensor = []
            for selected_frame_idx, decoded_frame in zip(selected_frames, decoded_frames):
                bbox = frame_bbox_dict[selected_frame_idx]
                image = Image.fromarray(decoded_frame).convert("RGB")

                # Use SAM2 to crop the background for classification
                if args.mask:
                    predictor.set_image(image)

                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox,
                        multimask_output=False,
                    )

                    instance_mask = masks[0]
                    mask = np.squeeze(instance_mask)
                else:
                    mask = None
                image = crop_object(np.array(image), bbox, mask)
                image_tensor.append(transform(image))
            
            image_tensor = torch.stack(image_tensor).to(device)

            with torch.no_grad():
                image_features = model(image_tensor).clone()

            cosine_similarity_dict = {}
            for query_feature in image_features:
                for char_name, feature_ls in character_bank.items():
                    cosine_sim = []

                    for feature in feature_ls:
                        cosine_sim.append(cosine_similarity(query_feature.detach().cpu().numpy(), feature))
                    cosine_sim.sort(reverse=True)

                    if len(cosine_sim) <= 3:
                        score = sum(cosine_sim) / len(cosine_sim)
                    else:
                        score = sum(cosine_sim[:3]) / 3

                    if char_name not in cosine_similarity_dict.keys():
                        cosine_similarity_dict[char_name] = score
                    else:
                        if cosine_similarity_dict[char_name] < score:
                            cosine_similarity_dict[char_name] = score
            
            def get_max_score_characters(cosine_similarity_dict):
                max_score = max(cosine_similarity_dict.values())
                max_score_characters = [name for name, score in cosine_similarity_dict.items() if score == max_score]
                return max_score, max_score_characters[0]

            max_score, character_name = get_max_score_characters(cosine_similarity_dict)
            classified_track = copy.deepcopy(unclassified_track)
            classified_track["label"] = character_name
            classified_track["score"] = float(max_score)

            classified_tracks.append(classified_track)

        classified_shot_results = copy.deepcopy(track)
        classified_shot_results["movie_title"] = current_movie_title
        classified_shot_results["year"] = os.path.basename(
            os.path.dirname(video_paths[current_video_idx])
        )
        classified_shot_results["clip_idx"] = current_video_idx
        classified_shot_results["tracks"] = classified_tracks

        classified_track_results.append(classified_shot_results)

    save_dir = os.path.dirname(os.path.abspath(args.save_file))
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_file, 'w') as outfile:
        json.dump(classified_track_results, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default='giant', type=str, help='Architecture')
    parser.add_argument('--sam2_checkpoint', default=None, type=str, help='Path to SAM2 checkpoint (required with --mask)')
    parser.add_argument('--source_dir', default=DEFAULT_SOURCE_DIR, type=str, help='Root directory containing source videos')
    parser.add_argument('--ckpt_dir', default=DEFAULT_CKPT_DIR, type=str, help='Path to finetuned DINOv2 weights directory')
    parser.add_argument('--char_feat_dir', default=DEFAULT_CHAR_FEAT_DIR, type=str, help='Path to character feature banks')
    parser.add_argument(
        '--movie_to_video_files',
        default=DEFAULT_MOVIE_TO_VIDEO_FILES,
        type=str,
        help='JSON mapping each movie title to its year/video-id entries',
    )
    parser.add_argument('--track_file', required=True, type=str)
    parser.add_argument('--save_file', required=True, type=str)
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()

    main()
