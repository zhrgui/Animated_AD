import argparse
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from PIL import Image
from glob import glob
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    if n <= 0:
        raise ValueError("Number of chunks must be positive")
    if not lst:
        return [[] for _ in range(n)]

    q, r = divmod(len(lst), n)
    chunks = []
    start = 0

    for i in range(n):
        end = start + q + (1 if i < r else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def get_chunk(lst, n, k):
    """Get the k-th chunk from n chunks"""
    chunks = split_list(lst, n)
    if k >= len(chunks):
        raise IndexError(f"Chunk index {k} is out of range for {len(chunks)} chunks.")
    return chunks[k]


class CharactersImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        self.label_set = sorted(
            list(set([file_name.split("_")[0] for file_name in os.listdir(img_dir)]))
        )
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
        self.label_to_indices = {label: [] for label in self.label_set}

        for file_name in os.listdir(img_dir):
            label = file_name.split("_")[0]
            if label in self.label_set:
                img_path = os.path.join(img_dir, file_name)
                self.img_paths.append(img_path)
                self.labels.append(self.label_to_idx[label])
                self.label_to_indices[label].append(len(self.img_paths) - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class AnchorPlusPerLabelSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.labels = dataset.label_set
        self.label_to_indices = dataset.label_to_indices
        self.num_labels = len(self.labels)
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        """Shuffle indices for each label if shuffle is enabled."""
        self.label_indices = {}
        for label in self.labels:
            indices = self.label_to_indices[label].copy()
            if self.shuffle:
                random.shuffle(indices)
            self.label_indices[label] = indices
        self.label_ptr = {label: 0 for label in self.labels}

    def __iter__(self):
        self.reset()

        for anchor_label in self.labels:
            if len(self.label_indices[anchor_label]) < 2:
                continue

            anchor_idx = self.label_indices[anchor_label][self.label_ptr[anchor_label]]
            self.label_ptr[anchor_label] += 1

            if self.label_ptr[anchor_label] >= len(self.label_indices[anchor_label]):
                if self.shuffle:
                    self.label_ptr[anchor_label] = 0
                    random.shuffle(self.label_indices[anchor_label])
                else:
                    print(f"No more positive samples for label '{anchor_label}'.")
                    continue

            positive_idx = self.label_indices[anchor_label][self.label_ptr[anchor_label]]
            self.label_ptr[anchor_label] += 1

            batch = [anchor_idx, positive_idx]

            for label in self.labels:
                if label == anchor_label:
                    continue

                if self.label_ptr[label] >= len(self.label_indices[label]) - 1:
                    if self.shuffle:
                        self.label_ptr[label] = 0
                        random.shuffle(self.label_indices[label])
                    else:
                        print(f"No more images for label '{label}'.")
                        continue

                neg_idx = self.label_indices[label][self.label_ptr[label]]
                self.label_ptr[label] += 1
                batch.append(neg_idx)
            yield batch

    def __len__(self):
        min_pairs = min(len(indices) // 2 for label, indices in self.label_to_indices.items() if len(indices) >= 2)
        return min_pairs


def nce_loss(batch_embeddings, positive_indices, temperature=0.07):
    """
    Computes the InfoNCE loss for a single anchor-positive pair within a batch.
    The anchor is contrasted against one positive and all other samples as negatives.
    """
    assert (len(positive_indices) == 2), "positive_indices must contain exactly two indices: [anchor_idx, positive_idx]."
    anchor_idx, positive_idx = positive_indices

    N = batch_embeddings.shape[0]
    assert (0 <= anchor_idx < N and 0 <= positive_idx < N), "positive_indices must be within the batch size."

    normalized_embeddings = F.normalize(batch_embeddings, dim=1)

    anchor = normalized_embeddings[anchor_idx].unsqueeze(0)
    positive = normalized_embeddings[positive_idx].unsqueeze(0)

    all_indices = torch.arange(N).to(batch_embeddings.device)
    negative_mask = ~torch.isin(all_indices, torch.tensor(positive_indices).to(batch_embeddings.device))
    negatives = normalized_embeddings[negative_mask]

    keys = torch.cat([positive, negatives], dim=0)
    logits = torch.matmul(anchor, keys.T) / temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(batch_embeddings.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def main(args):
    image_dir_list = glob(os.path.join(args.img_dir, "*"))

    # Test-time finetune DINOv2 for each movies 
    for image_dir in get_chunk(image_dir_list, args.num_chunks, args.chunk_idx):
        movie_title = image_dir.split("/")[-1]

        # Load the batches with one positive sample and other negative samples
        dataset = CharactersImageDataset(
            img_dir=image_dir,
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        batch_sampler = AnchorPlusPerLabelSampler(dataset)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained DINOv2 model
        model_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }

        model_arch = model_archs[args.model_size]
        model_name = f"dinov2_{model_arch}_reg"

        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, verbose=False)
        model.eval()
        model.to(device)

        # Set the last linear layer trainable
        for param in model.parameters():
            param.requires_grad = False

        last_block = model.blocks[-1]
        for param in last_block.mlp.w3.parameters():
            param.requires_grad = True

        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            for param in model.head.parameters():
                param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

        epoch_losses = []
        batch_losses = []

        for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
            model.train()
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                positive_indices = [i for i, x in enumerate(labels) if x == Counter(labels).most_common(1)[0][0]]
                optimizer.zero_grad()

                embeddings = model(images)
                loss = nce_loss(embeddings, positive_indices, temperature=args.temperature)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_losses.append(loss.item())

            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")
            scheduler.step()

            if (epoch + 1) % args.save_interval == 0:
                os.makedirs(os.path.join(args.save_dir, movie_title), exist_ok=False)
                torch.save(model.state_dict(), os.path.join(args.save_dir, movie_title, f"finetuned_dinov2_weights.pth"),)
        print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--save_dir", default=None, type=str, help="Path to save checkpoints.")
    parser.add_argument("--model_size", type=str, default="giant")
    parser.add_argument("--epochs", default=75, type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", default=6e-4, type=float, help="Learning rate for optimizer")
    parser.add_argument("--temperature", default=0.07, type=float, help="Temperature parameter for NCE loss")
    parser.add_argument( "--save_interval", default=75, type=int, help="How many epochs to wait before saving model checkpoint")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
