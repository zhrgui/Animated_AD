"""Filter retrieved character images down to at most one instance crop per image.

Every movie folder under --img_dir holds, per character, an optional anchor
image (<name>_0.<ext>, the wiki portrait) and retrieved images (<name>_k.<ext>,
k >= 1). Each retrieved image is run through OWLv2 to propose character boxes,
and DINOv2 embeds the box crop of each candidate. With --mask, SAM2 additionally
masks each box and the off-mask background is whited out; by default crops keep
the full box content, since SAM masks are sometimes incomplete.

With an anchor, the instance most similar to the anchor embedding is kept for
each retrieved image. An instance only counts for a character if that
character's anchor is its best match among all the movie's anchors
(--disambig_margin), which stops a co-starring character's crop from being
credited to the wrong bank entry. Without an anchor, the instances pooled over
all of the character's retrieved images are clustered (agglomerative, cosine
distance) and the largest cluster's centroid plays the anchor's role.

Selection is quality-first with a quota: images pass outright at
--sim_threshold, but the best --min_keep images are kept anyway as long as
they clear --floor_threshold, so sparse characters still get examples.
Detection-side filters drop group boxes that contain two or more other
detections, boxes smaller than --min_box_size, and crops blurrier than
--min_sharpness. At most one crop per retrieved image survives; crops are
saved under --save_dir/<movie>/ with their original filenames.
"""
import os
import re
import argparse
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as Image
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import binary_fill_holes, binary_erosion, laplace
from sklearn.cluster import AgglomerativeClustering
from transformers import Owlv2Processor, Owlv2ForObjectDetection

FNAME_RE = re.compile(r'^(?P<name>.+)_(?P<idx>\d+)\.(?P<ext>jpg|jpeg|png)$', re.IGNORECASE)


def rgba_to_rgb_alpha_to_white(image, alpha_threshold=100):
    if image.mode != 'RGBA':
        return image.convert("RGB"), False
    image_array = np.array(image)
    rgb_array = image_array[:, :, :3]
    alpha_channel = image_array[:, :, 3]
    alpha_mask = alpha_channel < alpha_threshold
    rgb_array[alpha_mask] = [255, 255, 255]
    return Image.fromarray(rgb_array), True


def crop_image(image, bbox, mask=None):
    """Crop the box out of the image, whiting out everything off-mask."""
    x_min, y_min, x_max, y_max = map(int, bbox)
    width, height = image.size
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width, x_max), min(height, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None

    if mask is None:
        return image.crop((x_min, y_min, x_max, y_max))

    np_image = np.array(image)
    cropped_image = np_image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    white_background = np.full_like(cropped_image, 255)

    mask_bool = cropped_mask.astype(bool)
    mask_bool = binary_erosion(mask_bool, iterations=2)
    mask_bool = binary_fill_holes(mask_bool)

    if cropped_image.ndim == 3 and cropped_image.shape[2] > 1:
        mask_bool = mask_bool[..., None]

    result_image = np.where(mask_bool, cropped_image, white_background)
    return Image.fromarray(result_image.astype(np.uint8))


def sharpness(crop):
    """Variance of the Laplacian at a fixed scale; low values mean blur.

    Calibrated on box crops: genuine blur/fog scores < 10 while even flat-shaded
    animation (plain white yeti fur) stays near 30, so a threshold around 15
    separates them.
    """
    gray = np.array(crop.convert("L").resize((224, 224)), dtype=np.float32)
    return float(laplace(gray).var())


def collect_characters(movie_dir):
    """Map character name -> {'anchor': path or None, 'retrieved': [paths]}."""
    characters = defaultdict(lambda: {"anchor": None, "retrieved": []})
    for fname in sorted(os.listdir(movie_dir)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        path = os.path.join(movie_dir, fname)
        entry = characters[m.group("name")]
        if int(m.group("idx")) == 0:
            entry["anchor"] = path
        else:
            entry["retrieved"].append(path)
    return characters


class InstanceExtractor:
    """OWLv2 boxes -> box crops (optionally SAM2-masked) -> DINOv2 embeddings."""

    def __init__(self, args, device):
        self.device = device
        self.box_threshold = args.box_threshold
        self.max_boxes = args.max_boxes
        self.use_mask = args.mask
        self.min_box_size = args.min_box_size
        self.min_sharpness = args.min_sharpness
        self.rel_box_ratio = args.rel_box_ratio

        print("initializing Owlv2")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-large-patch14-ensemble").to(device)
        # The second query exists only to claim body-part boxes (a paw filling
        # the frame scores high against its own character's anchor, so DINO
        # similarity cannot reject it); boxes it wins are discarded.
        self.texts = [["a photo of an animated character",
                       "a close-up of a hand or paw"]]

        if self.use_mask:
            print("initializing sam2")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(model_cfg, args.sam2_checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)

        print("initializing dinov2")
        model_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"}
        model_name = f"dinov2_{model_archs[args.model_size]}_reg"
        if args.pretrained_weights is None:
            self.dino = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name)
        else:
            self.dino = torch.hub.load(repo_or_dir="facebookresearch/dinov2",
                                       model=model_name, pretrained=False)
            state_dict = torch.load(args.pretrained_weights, map_location=device, weights_only=True)
            self.dino.load_state_dict(state_dict)
        self.dino.eval()
        self.dino.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def detect_boxes(self, image):
        """Return up to max_boxes boxes sorted by descending confidence.

        Boxes smaller than min_box_size on their short side are dropped, as are
        group boxes: a box that mostly contains two or more other detections is
        a crowd shot, not a single character instance.
        """
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt",
                                padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        target_sizes = torch.Tensor([image.size]).to(self.device)
        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.box_threshold)
        keep = results[0]["labels"].detach().cpu().numpy() == 0
        boxes = results[0]["boxes"].detach().cpu().numpy()[keep]
        scores = results[0]["scores"].detach().cpu().numpy()[keep]
        order = np.argsort(scores)[::-1]
        boxes = boxes[order].tolist()
        scores = scores[order].tolist()

        # Marginal detections in an image that also has a confident one are
        # usually body parts or fragments (a furry hand can out-DINO the real
        # character), so score them relative to the image's best box.
        if scores:
            boxes = [b for b, s in zip(boxes, scores)
                     if s >= self.rel_box_ratio * scores[0]]

        def containment(outer, inner):
            ix = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
            iy = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
            inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
            return ix * iy / inner_area if inner_area > 0 else 0.0

        big_enough = [b for b in boxes
                      if min(b[2] - b[0], b[3] - b[1]) >= self.min_box_size]
        kept = [b for b in big_enough
                if sum(1 for o in big_enough
                       if o is not b and containment(b, o) > 0.8) < 2]
        return kept[:self.max_boxes]

    def segment(self, image, boxes):
        """One SAM2 mask per box, prompting on the shared image encoding."""
        self.predictor.set_image(image)
        masks = []
        for bbox in boxes:
            mask, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None, box=[bbox], multimask_output=False)
            masks.append(np.squeeze(mask[0]))
        return masks

    def embed(self, crops):
        """L2-normalized DINOv2 CLS embeddings, one row per crop."""
        batch = torch.stack([self.transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            feats = self.dino(batch)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy()

    def anchor_feature(self, anchor_path):
        """Embed the anchor, cropped to its main instance like a profile image."""
        image = Image.open(anchor_path)
        image, is_clean = rgba_to_rgb_alpha_to_white(image)
        boxes = self.detect_boxes(image)
        if boxes:
            # A wiki portrait shows one character, so the largest box is the
            # character; the most confident one is often just their face, and
            # a face-only anchor embedding mismatches every full-body crop.
            bbox = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            # RGBA anchors are already stylized cut-outs; skip masking for them.
            mask = self.segment(image, [bbox])[0] if self.use_mask and not is_clean else None
            cropped = crop_image(image, bbox, mask)
            if cropped is not None:
                image = cropped
        return self.embed([image])[0]

    def image_instances(self, image_path):
        """All candidate instances of one retrieved image: (crop, feature) pairs."""
        try:
            image = Image.open(image_path)
            image, _ = rgba_to_rgb_alpha_to_white(image)
        except Exception:
            return []
        boxes = self.detect_boxes(image)
        if not boxes:
            return []
        masks = self.segment(image, boxes) if self.use_mask else [None] * len(boxes)
        crops = []
        for bbox, mask in zip(boxes, masks):
            cropped = crop_image(image, bbox, mask)
            if cropped is not None and sharpness(cropped) >= self.min_sharpness:
                crops.append(cropped)
        if not crops:
            return []
        feats = self.embed(crops)
        return list(zip(crops, feats))


def keep_with_quota(scored, sim_threshold, floor_threshold, min_keep):
    """Keep everything above sim_threshold, then fill up to min_keep with the
    next-best images that still clear floor_threshold."""
    selected = {}
    for sim, path, crop in sorted(scored, key=lambda t: t[0], reverse=True):
        if sim >= sim_threshold or (len(selected) < min_keep and sim >= floor_threshold):
            selected[path] = crop
    return selected


def select_by_anchors(extractor, anchored, anchor_feats, args):
    """Two-round selection for all of a movie's anchored characters at once.

    Round 1 scores every instance against the wiki anchors. Wiki anchors are
    often stylistically far from in-film renders, so round 2 rescores against
    prototypes: each character's anchor averaged with the features of their
    confident round-1 picks. In-film look-alikes (two kids the anchor barely
    separates) resolve in round 2 because each prototype now speaks the film's
    own style. In both rounds an instance is skipped when some other
    character's representation explains it better (--disambig_margin).
    """
    instances = {name: {path: extractor.image_instances(path)
                        for path in entry["retrieved"]}
                 for name, entry in tqdm(anchored.items(), desc="instances")}

    def run_round(feats_bank, disambiguate=True):
        picks = {}
        for name in instances:
            own_feat = feats_bank[name]
            other_feats = [f for n, f in feats_bank.items() if n != name]
            scored = []
            for path, path_instances in instances[name].items():
                best = None
                for crop, feat in path_instances:
                    own_sim = float(np.dot(feat, own_feat))
                    other_sim = max((float(np.dot(feat, f)) for f in other_feats),
                                    default=-1.0)
                    if disambiguate and own_sim + args.disambig_margin < other_sim:
                        continue
                    if best is None or own_sim > best[0]:
                        best = (own_sim, crop, feat)
                if best is not None:
                    scored.append((best[0], path, best[1], best[2]))
            picks[name] = scored
        return picks

    round1 = run_round(anchor_feats)
    prototypes = {}
    for name, scored in round1.items():
        confident = [feat for sim, _, _, feat in scored if sim >= args.sim_threshold]
        proto = np.mean([anchor_feats[name]] + confident, axis=0)
        prototypes[name] = proto / np.linalg.norm(proto)
    round2 = run_round(prototypes)

    selections = {name: keep_with_quota([(sim, path, crop) for sim, path, crop, _ in scored],
                                        args.sim_threshold, args.floor_threshold, args.min_keep)
                  for name, scored in round2.items()}

    # A character whose every candidate lost the cross-character vote would end
    # up with no examples at all. For those, retry without the veto so their
    # best plausible crop comes back — but fill only a single image, since
    # without the veto there is nothing keeping look-alikes out, and flag the
    # character for manual review.
    empty = [name for name, sel in selections.items() if not sel]
    if empty:
        rescue = run_round(prototypes, disambiguate=False)
        for name in empty:
            scored = sorted(((sim, path, crop) for sim, path, crop, _ in rescue[name]),
                            key=lambda t: t[0], reverse=True)
            if scored and scored[0][0] >= args.floor_threshold:
                _, path, crop = scored[0]
                selections[name] = {path: crop}
                print(f"  [rescued] {name}: kept 1 image without "
                      f"cross-character veto; review manually")
    return selections


def select_by_clustering(extractor, retrieved_paths, args):
    """Cluster all instances, then let the largest cluster's centroid play the
    anchor's role: score every image's best instance against it and apply the
    same quality-with-quota keep rule as the anchor path."""
    sources, crops, feats = [], [], []
    for path in retrieved_paths:
        for crop, feat in extractor.image_instances(path):
            sources.append(path)
            crops.append(crop)
            feats.append(feat)
    if not crops:
        return {}
    feats = np.stack(feats)

    if len(crops) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=args.cluster_threshold)
        labels = clustering.fit_predict(feats)

    # The dominant character should recur across images, so rank clusters by
    # how many distinct retrieved images they cover, not by raw instance count.
    coverage = {label: len({sources[i] for i in np.flatnonzero(labels == label)})
                for label in np.unique(labels)}
    best_label = max(coverage, key=coverage.get)
    if coverage[best_label] < args.min_cluster_size:
        return {}

    member_idx = np.flatnonzero(labels == best_label)
    centroid = feats[member_idx].mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    best_per_image = {}  # path -> (similarity to centroid, crop)
    for i, path in enumerate(sources):
        sim = float(np.dot(feats[i], centroid))
        if path not in best_per_image or sim > best_per_image[path][0]:
            best_per_image[path] = (sim, crops[i])
    scored = [(sim, path, crop) for path, (sim, crop) in best_per_image.items()]
    # An instance within cluster_threshold cosine distance of the centroid would
    # have joined the cluster, so that distance doubles as the pass bar here.
    return keep_with_quota(scored, 1.0 - args.cluster_threshold,
                           args.floor_threshold, args.min_keep)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = InstanceExtractor(args, device)

    movies = sorted(d for d in os.listdir(args.img_dir)
                    if os.path.isdir(os.path.join(args.img_dir, d)))
    if args.movie is not None:
        movies = [m for m in movies if m == args.movie]

    for movie in movies:
        movie_dir = os.path.join(args.img_dir, movie)
        out_dir = os.path.join(args.save_dir, movie)
        os.makedirs(out_dir, exist_ok=True)
        if not args.overwrite and os.listdir(out_dir):
            print(f"[{movie}] output exists, skipping (use --overwrite to redo)")
            continue
        characters = collect_characters(movie_dir)

        # Anchor features for every character up front: matching an instance
        # against all of a movie's anchors is what lets us reject crops that a
        # co-starring character explains better.
        anchor_feats = {name: extractor.anchor_feature(entry["anchor"])
                        for name, entry in tqdm(characters.items(), desc=f"{movie} anchors")
                        if entry["anchor"] is not None}

        anchored = {name: entry for name, entry in characters.items()
                    if name in anchor_feats and entry["retrieved"]}
        selections = select_by_anchors(extractor, anchored, anchor_feats, args) \
            if anchored else {}
        for name, entry in characters.items():
            if entry["retrieved"] and name not in anchor_feats:
                selections[name] = select_by_clustering(extractor, entry["retrieved"], args)

        for name, selected in selections.items():
            for src_path, crop in selected.items():
                crop.save(os.path.join(out_dir, os.path.basename(src_path)))
            mode = "anchor" if name in anchor_feats else "cluster"
            print(f"[{movie}] {name}: kept "
                  f"{len(selected)}/{len(characters[name]['retrieved'])} ({mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/work/zhongrui/datasets/AnimatedAD_data/src/images", type=str)
    parser.add_argument('--save_dir', default="/work/zhongrui/datasets/AnimatedAD_data/charbank/appearance", type=str)
    parser.add_argument('--mask', action='store_true',
                        help="White out the background with SAM2 masks before cropping")
    parser.add_argument('--sam2_checkpoint', default=None, type=str,
                        help="Path to SAM2 checkpoint (required if --mask is set)")
    parser.add_argument('--pretrained_weights', default=None, type=str,
                        help="Optional finetuned DINOv2 state dict")
    parser.add_argument('--model_size', default='giant', type=str,
                        choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--box_threshold', default=0.1, type=float,
                        help="OWLv2 detection confidence threshold")
    parser.add_argument('--max_boxes', default=5, type=int,
                        help="Max candidate instances per retrieved image")
    parser.add_argument('--sim_threshold', default=0.5, type=float,
                        help="Cosine similarity to the anchor above which an image passes outright")
    parser.add_argument('--floor_threshold', default=0.35, type=float,
                        help="Absolute minimum similarity, even when filling the min_keep quota")
    parser.add_argument('--rel_box_ratio', default=0.5, type=float,
                        help="Drop boxes scoring below this fraction of the image's best box")
    parser.add_argument('--min_keep', default=5, type=int,
                        help="Target number of images per character the quota fill aims for")
    parser.add_argument('--disambig_margin', default=0.1, type=float,
                        help="Slack when requiring the own anchor to be an instance's best match")
    parser.add_argument('--min_box_size', default=48, type=int,
                        help="Min short side of a detection box, in pixels")
    parser.add_argument('--min_sharpness', default=15.0, type=float,
                        help="Min Laplacian variance of a crop; genuine blur scores below ~10")
    parser.add_argument('--cluster_threshold', default=0.3, type=float,
                        help="Cosine distance threshold for agglomerative clustering (no-anchor case)")
    parser.add_argument('--min_cluster_size', default=2, type=int,
                        help="Min distinct images the winning cluster must cover")
    parser.add_argument('--movie', default=None, type=str,
                        help="Process only this movie folder")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-process characters that already have crops saved")
    args = parser.parse_args()

    main(args)
