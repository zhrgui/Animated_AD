import os
import argparse
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import crop_image

def rgba_to_rgb_alpha_to_white(image, alpha_threshold=100):
    if image.mode != 'RGBA':
        return image.convert("RGB"), False
    image_array = np.array(image)
    rgb_array = image_array[:, :, :3]
    alpha_channel = image_array[:, :, 3]
    alpha_mask = alpha_channel < alpha_threshold
    rgb_array[alpha_mask] = [255, 255, 255]
    return Image.fromarray(rgb_array), True

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
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    dot_product = np.dot(arr1, arr2)
    magnitude_arr1 = np.linalg.norm(arr1)
    magnitude_arr2 = np.linalg.norm(arr2)

    if magnitude_arr1 == 0 or magnitude_arr2 == 0:
        return 0.0
    cosine_sim = dot_product / (magnitude_arr1 * magnitude_arr2)
    return cosine_sim

def top_k_indices(lst, k):
    indices = np.argsort(lst)[-k:][::-1]
    return indices

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load SAM2 and Owlv2
    if args.mask:
        print("initializing sam2")
        sam2_checkpoint = args.sam2_checkpoint
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

    if args.box:
        print("initializing Owlv2")
        processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
        owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(device)
        texts = [["a photo of an animated character"]]

    # Load test-time finetuned DINOv2
    print("initializing dinov2")
    model_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    model_arch = model_archs[args.model_size]
    model_name = f"dinov2_{model_arch}_reg"

    if args.pretrained_weights is None:
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name)
    else:
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, pretrained=False)
        state_dict = torch.load(args.pretrained_weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if args.vis_folder is not None:
        os.makedirs(args.vis_folder, exist_ok=True)

    # Separate profile images and retrieval images for filtering. With
    # --embed_all every image is treated like a profile image (embedded as-is):
    # used for pre-cropped instance banks, where selection already happened and
    # there are no _0 anchors.
    profile_images = []
    retrieval_images = []
    for image_file in os.listdir(args.img_dir):
        if args.embed_all or os.path.basename(image_file).split(".")[0].split("_")[-1] == "0":
            profile_images.append(image_file)
        else:
            retrieval_images.append(image_file)

    # Processing the profile images first
    for image_file in tqdm(profile_images, total=len(profile_images)):
        # If the image is in RGBA mode, convert it into RGB
        image = Image.open(os.path.join(args.img_dir, image_file))
        image, is_clean = rgba_to_rgb_alpha_to_white(image)

        # Visualize the selected embeddings as a feature bank
        if args.vis_folder is not None:
            output_filename = os.path.join(args.vis_folder, image_file)
        else:
            output_filename = None

        if args.box:
            inputs = processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target_sizes = torch.Tensor([image.size]).to(device)

            with torch.no_grad():
                outputs = owlv2_model(**inputs)
                results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
                input_boxes = results[0]["boxes"].detach().cpu().numpy().tolist()

            # If more than one boxes detected, use the most confident one
            if len(input_boxes) > 1:
                idx = torch.argmax(results[0]["scores"])
                bbox = input_boxes[idx]
                use_box_prompts = True

            # If only one box detected, use it directly
            elif len(input_boxes) == 1:
                bbox = input_boxes[0]
                use_box_prompts = True

            # If no boxes detected, do not use any box as prompt
            else:
                width, height = image.size
                bbox = [0, 0, width, height]
                use_box_prompts = False

            if args.mask:
                # Not RGBA with boxes detected
                if not is_clean and use_box_prompts:
                    predictor.set_image(image)

                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=[bbox],
                        multimask_output=False,
                    )
                    instance_mask = np.squeeze(masks[0])

                # Not RGBA with no boxes detected
                elif not is_clean and not use_box_prompts:
                    predictor.set_image(image)

                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=None,
                        multimask_output=False,
                    )

                    instance_mask = np.array(masks[0])
                
                # Skip SAM2 for RGBA images, as they are typically stylized profile images that do not require cropping
                else:
                    instance_mask = None
            else:
                instance_mask = None

            # Crop the image with the resulted box and mask
            image = crop_image(image, bbox, instance_mask, output_filename)

        # Use the CLS token from DINOv2 as the feature representation for character images
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(image_tensor).squeeze().clone()
        feats_np = feats.detach().cpu().numpy()
        if image_file.lower().endswith(".jpg"):
            image_file = image_file.replace('.jpg', '.npz')
        elif image_file.lower().endswith(".png"):
            image_file = image_file.replace('.png', '.npz')
        os.makedirs(args.save_folder, exist_ok=True)
        save_filename = os.path.join(args.save_folder, image_file)
        np.savez(save_filename, feature=feats_np)
    
    if args.embed_all:
        return

    # Load the character bank from the profile images
    character_bank = load_character_bank(args.save_folder)

    # Processing the retrieved images and filtering the instances
    character_selections = {}
    for character_name in character_bank:
        character_selections[character_name] = {"image_name": [], "feature": [], "cos_similarity": []}

    for image_file in tqdm(retrieval_images, total=len(retrieval_images)):
        # Error handling when sometimes the image downloaded fails
        try:
            image = Image.open(os.path.join(args.img_dir, image_file))
            image, _ = rgba_to_rgb_alpha_to_white(image)
        except:
            continue

        if args.vis_folder is not None:
            output_filename = os.path.join(args.vis_folder, image_file)
        else:
            output_filename = None

        # Call OWLv2 for character box detection
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.Tensor([image.size]).to(device)
        
        with torch.no_grad():
            outputs = owlv2_model(**inputs)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            input_boxes = results[0]["boxes"].detach().cpu().numpy().tolist()

        if len(input_boxes) > 1:
            # Optionally, use all boxes or the most confident one
            idx = torch.argmax(results[0]["scores"])
            bbox = input_boxes[idx]
        elif len(input_boxes) == 1:
            bbox = input_boxes[0]
        else:
            continue

        # Call SAM2 for instance mask given the detected box as prompt
        predictor.set_image(image)

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=[bbox],
            multimask_output=False,
        )

        instance_mask = masks[0]

        image = crop_image(image, bbox, np.squeeze(instance_mask), output_filename)

        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(image_tensor).squeeze().clone()
        feats_np = feats.detach().cpu().numpy()

        suffix = "_" + image_file.split("_")[-1]
        character_name = image_file.replace(suffix, "")

        # Some characters have retrieval images but no profile image, because
        # their wiki page carries no usable portrait. They are listed in the
        # character table with an "Anchor confidence" of "missing". There is no
        # anchor to rank their images against, so skip them here rather than
        # raise; they need an anchor before they can be filtered.
        if character_name not in character_bank:
            continue
        profile_image_feautre = character_bank[character_name]

        character_selections[character_name]["image_name"].append(image_file)
        character_selections[character_name]["feature"].append(feats_np)
        character_selections[character_name]["cos_similarity"].append(cosine_similarity(feats_np, profile_image_feautre))

    # Select the most possible k instance features for each character based on feature similarity
    for character_name, retrieved_instances in character_selections.items():
        image_ls = retrieved_instances["image_name"]
        feature_ls = retrieved_instances["feature"]
        cos_similarities = retrieved_instances["cos_similarity"]

        instance_indices = top_k_indices(cos_similarities, k=args.num_per_character)

        selected_image_ls = [image_ls[idx] for idx in instance_indices]
        selected_feature_ls = [feature_ls[idx] for idx in instance_indices]

        for image_file, feats_np in zip(selected_image_ls, selected_feature_ls):
            if image_file.lower().endswith(".jpg"):
                image_file = image_file.replace('.jpg', '.npz')
            elif image_file.lower().endswith(".png"):
                image_file = image_file.replace('.png', '.npz')
            save_filename = os.path.join(args.save_folder, image_file)
            np.savez(save_filename, feature=feats_np)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default=None, type=str)
    parser.add_argument('--save_folder', default=None, type=str)
    parser.add_argument('--pretrained_weights', default=None, type=str)
    parser.add_argument('--model_size', default='giant', type=str, help='Architecture')
    parser.add_argument('--num_per_character', default=8, type=int)
    parser.add_argument('--vis_folder', default=None, type=str)

    parser.add_argument('--embed_all', action='store_true',
                        help="Embed every image directly (no profile/retrieval split, "
                             "no top-k filtering); for pre-cropped instance images")
    parser.add_argument('--mask', action='store_true', help="Enable masking")
    parser.add_argument('--box', action='store_true', help="Enable bounding box detection")
    parser.add_argument('--sam2_checkpoint', default=None, type=str, help='Path to SAM2 checkpoint (required if --mask is set)')
    args = parser.parse_args()

    main()
