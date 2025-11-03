import argparse
import copy
import os
import json
import random
import numpy as np
import torch
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, binary_erosion

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

VIDEO_IDX_TO_MOVIE_TITLES = ""

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
        assert AssertionError("Character bank doesn't exist")
    character_bank = {}
    for file in os.listdir(character_bank_dir):
        char_name = os.path.splitext(file)[0].split("_")[0]
        char_feature = np.load(os.path.join(character_bank_dir, file))['feature']
        character_bank[char_name] = char_feature
    return character_bank

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
    # Load the mapping between video indices to the corresponding movie titles
    video_idx_to_movie_titles_mapping = load_json(VIDEO_IDX_TO_MOVIE_TITLES)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Intializing SAM2")
    sam2_checkpoint = "/scratch/shared/beegfs/zhongrui/sam2_ckpt/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print("initializing dinov2")
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
    
    last_movie_title = None

    classified_track_results = []
    for track in unclassified_track_results:
        current_video_idx = track["video_idx"]
        current_movie_title = video_idx_to_movie_titles_mapping[current_video_idx]

        # Avoid redundant model loading
        if last_movie_title != current_movie_title:
            # Load test-time finetuned DINOv2 weights
            pretrained_weights = os.path.join(args.ckpt_dir, current_movie_title, "finetuned_dinov2_weights.pth")
            model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, pretrained=False)
            state_dict = torch.load(pretrained_weights, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            # Load the character feature bank for the movie
            character_bank = load_character_bank(os.path.join(args.char_feat_dir, current_movie_title))

            last_movie_title = current_movie_title
        else:
            continue

        k = 5
        classified_tracks = []
        shot_folder = os.path.join(args.frame_dir, current_video_idx)
        for unclassified_track in track["track"]:
            # Randomly select frames across the track for character recognition
            frame_bbox_dict = unclassified_track["track"]
            frame_idx_ls = list(frame_bbox_dict.keys())
            if k < len(frame_idx_ls):
                selected_frames = random.sample(frame_idx_ls, k)
            else:
                selected_frames = frame_idx_ls

            image_tensor = []
            for selected_frame_idx in selected_frames:
                image_path = os.path.join(shot_folder, track["shot_idx"], f"{selected_frame_idx}.jpg") 
                bbox = frame_bbox_dict[selected_frame_idx]
                image = Image.open(image_path).convert("RGB")

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
        del classified_shot_results["track"]
        classified_shot_results["track"] = classified_tracks

        classified_track_results.append(classified_shot_results)

    with open(args.save_file, 'w') as outfile:
        json.dump(classified_track_results, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default='giant', type=str, help='Architecture')
    parser.add_argument('--ckpt_dir', default="/scratch/shared/beegfs/zhongrui/cmd_character_bank/finetuned_dinov2_weights", type=str)
    parser.add_argument('--track_file', default=None, type=str)
    parser.add_argument('--frame_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()

    main()
