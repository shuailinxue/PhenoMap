"""Cache-aware KEEP and source-prompt-tuned PathPT query baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from step3.data.dataset import _load_he_image


def image_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def grid_coordinates(height: int, width: int, patch_size: int, stride: int):
    half = patch_size // 2
    if height < patch_size or width < patch_size:
        return [(width // 2, height // 2)]
    x_values = list(range(half, width - half + 1, stride)) or [width // 2]
    y_values = list(range(half, height - half + 1, stride)) or [height // 2]
    if x_values[-1] != width - half:
        x_values.append(width - half)
    if y_values[-1] != height - half:
        y_values.append(height - half)
    return [(x, y) for y in y_values for x in x_values]


def extract_patch(image: np.ndarray, x: int, y: int, patch_size: int):
    half = patch_size // 2
    height, width = image.shape[:2]
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half
    patch = image[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    padding = (
        (max(0, -y0), max(0, y1 - height)),
        (max(0, -x0), max(0, x1 - width)),
        (0, 0),
    )
    if any(value for pair in padding[:2] for value in pair):
        mode = "reflect" if min(patch.shape[:2]) > 1 else "edge"
        patch = np.pad(patch, padding, mode=mode)
    return patch


def is_tissue(patch: np.ndarray, threshold: float):
    rgb = patch.astype(np.float32)
    brightness = rgb.mean(axis=-1)
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    return float(((brightness < 230) & (saturation > 15)).mean()) >= threshold


class QueryPatchDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        patch_size: int,
        stride: int,
        tissue_threshold: float,
    ):
        self.image = image
        self.patch_size = patch_size
        self.coordinates = [
            coordinate
            for coordinate in grid_coordinates(*image.shape[:2], patch_size, stride)
            if is_tissue(
                extract_patch(image, *coordinate, patch_size), tissue_threshold
            )
        ]
        self.transform = image_transform()

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        x, y = self.coordinates[index]
        patch = extract_patch(self.image, x, y, self.patch_size)
        return self.transform(patch), torch.tensor([float(x), float(y)])


class PromptLearner(nn.Module):
    def __init__(self, model, tokenizer, device, n_context=16):
        super().__init__()
        class_names = ["normal tissue", "tumor tissue"]
        context_words = "a histopathology image of".split()
        context_words = (
            context_words * ((n_context + len(context_words) - 1) // len(context_words))
        )[:n_context]
        context_text = " ".join(context_words)
        tokenized_context = tokenizer(
            [context_text] * len(class_names),
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            context = model.text.get_input_embeddings()(
                tokenized_context["input_ids"].to(device)
            )[:, 1 : 1 + n_context]
        self.context = nn.Parameter(context)

        placeholder = " ".join(["X"] * n_context)
        prompts = [f"{placeholder} {name}." for name in class_names]
        self.tokenized = tokenizer(
            prompts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = model.text.get_input_embeddings()(
                self.tokenized["input_ids"].to(device)
            )
        self.register_buffer("prefix", embeddings[:, :1])
        self.register_buffer("suffix", embeddings[:, 1 + n_context :])

    def forward(self):
        return torch.cat([self.prefix, self.context, self.suffix], dim=1)


class SupportPatchDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        patch_size: int,
        patches_per_class: int,
        seed: int,
    ):
        if mask.shape != image.shape[:2]:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                )
            )
        positive = mask == 1
        groups = {0: [], 1: []}
        half = patch_size // 2
        for x, y in grid_coordinates(*image.shape[:2], patch_size, patch_size):
            fraction = positive[y - half : y + half, x - half : x + half].mean()
            groups[int(fraction >= 0.25)].append((x, y))
        count = min(patches_per_class, len(groups[0]), len(groups[1]))
        if count == 0:
            raise ValueError("The HBC1 support image needs positive and negative patches")
        rng = np.random.default_rng(seed)
        self.samples = []
        for label in [0, 1]:
            chosen = rng.choice(len(groups[label]), count, replace=False)
            self.samples.extend([(*groups[label][index], label) for index in chosen])
        self.image = image
        self.patch_size = patch_size
        self.transform = image_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, label = self.samples[index]
        patch = extract_patch(self.image, x, y, self.patch_size)
        return self.transform(patch), torch.tensor(label, dtype=torch.long)


def encode_prompt_features(model, prompt_learner, device):
    prompts = prompt_learner()
    attention = prompt_learner.tokenized["attention_mask"].to(device)
    output = model.text(inputs_embeds=prompts, attention_mask=attention)
    return F.normalize(output.pooler_output, dim=-1)


def train_pathpt_prompt(
    model,
    prompt_learner,
    loader,
    device,
    *,
    steps=200,
    learning_rate=1e-3,
):
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=learning_rate)
    completed = 0
    while completed < steps:
        for patches, labels in loader:
            if completed >= steps:
                break
            with torch.no_grad():
                image_features = model.encode_image(patches.to(device))
            text_features = encode_prompt_features(model, prompt_learner, device)
            logits = image_features @ text_features.T * model.logit_scale.exp()
            loss = F.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed += 1


def keep_text_features(model, tokenizer, device):
    tumor_names = [
        "tumor tissue",
        "tumor epithelial tissue",
        "cancerous tissue",
        "breast tumor tissue",
        "breast tumor epithelial tissue",
        "breast cancerous tissue",
    ]
    normal_names = [
        "normal tissue",
        "non-cancerous tissue",
        "normal breast tissue",
        "breast non-cancerous tissue",
        "benign breast tissue",
        "benign tissue",
    ]
    templates = [
        "CLASSNAME.",
        "a photomicrograph showing CLASSNAME.",
        "a histopathological image of CLASSNAME.",
        "an H&E stained image of CLASSNAME.",
        "an image of CLASSNAME.",
        "presence of CLASSNAME.",
    ]

    def encode(names):
        prompts = [template.replace("CLASSNAME", name) for name in names for template in templates]
        tokens = tokenizer(
            prompts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        return F.normalize(model.encode_text(tokens).mean(dim=0, keepdim=True), dim=-1)

    return torch.cat([encode(normal_names), encode(tumor_names)])


@torch.no_grad()
def infer_scores(model, text_features, loader, device):
    scores = []
    coordinates = []
    for patches, batch_coordinates in loader:
        features = model.encode_image(patches.to(device))
        logits = features @ text_features.T * model.logit_scale.exp()
        scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        coordinates.extend(batch_coordinates.numpy().tolist())
    return np.concatenate(scores), coordinates


def save_score_map(
    image,
    scores,
    coordinates,
    output_path,
    *,
    patch_size,
    scale=0.25,
    smooth_sigma=112,
):
    height, width = image.shape[:2]
    out_height, out_width = int(height * scale), int(width * scale)
    score_sum = np.zeros((out_height, out_width), dtype=np.float32)
    count = np.zeros_like(score_sum)
    half = patch_size // 2
    for score, (x, y) in zip(scores, coordinates):
        y0 = max(0, int(np.floor((y - half) * scale)))
        y1 = min(out_height, int(np.ceil((y + half) * scale)))
        x0 = max(0, int(np.floor((x - half) * scale)))
        x1 = min(out_width, int(np.ceil((x + half) * scale)))
        score_sum[y0:y1, x0:x1] += score
        count[y0:y1, x0:x1] += 1
    valid = count > 0
    score_map = np.zeros_like(score_sum)
    np.divide(score_sum, count, out=score_map, where=valid)
    score_map = gaussian_filter(score_map, sigma=smooth_sigma * scale)
    score_map[~valid] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, score_map.astype(np.float16))
    pd.DataFrame(
        {"x": [item[0] for item in coordinates], "y": [item[1] for item in coordinates], "query_score": scores}
    ).to_csv(output_path.with_name(output_path.name.replace("score_map.npy", "patch_scores.csv")), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a morphology-query baseline")
    parser.add_argument("--mode", choices=["keep", "pathpt"], required=True)
    parser.add_argument("--query-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-image", type=Path)
    parser.add_argument("--support-mask", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--tissue-threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patches-per-class", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        print("Cached score map found; no baseline inference was run.")
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        "Astaxanthin/KEEP", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "Astaxanthin/KEEP", trust_remote_code=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if args.mode == "keep":
        text_features = keep_text_features(model, tokenizer, device)
    else:
        if args.support_image is None or args.support_mask is None:
            raise ValueError("PathPT requires HBC1 support image and mask")
        prompt = PromptLearner(model, tokenizer, device).to(device)
        prompt_cache = args.output.with_name("prompt_learner_HBC1_support.pt")
        if prompt_cache.exists():
            prompt.load_state_dict(torch.load(prompt_cache, map_location=device))
        else:
            support_image = _load_he_image(str(args.support_image))
            support_mask = tifffile.imread(args.support_mask)
            support_data = SupportPatchDataset(
                support_image,
                support_mask,
                args.patch_size,
                args.patches_per_class,
                args.seed,
            )
            support_loader = DataLoader(support_data, batch_size=64, shuffle=True)
            train_pathpt_prompt(model, prompt, support_loader, device, steps=args.train_steps)
            prompt_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prompt.state_dict(), prompt_cache)
        text_features = encode_prompt_features(model, prompt, device)

    query_image = _load_he_image(str(args.query_image))
    query_data = QueryPatchDataset(
        query_image, args.patch_size, args.stride, args.tissue_threshold
    )
    loader = DataLoader(query_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scores, coordinates = infer_scores(model, text_features, loader, device)
    save_score_map(
        query_image,
        scores,
        coordinates,
        args.output,
        patch_size=args.patch_size,
    )
    print("Baseline score map generated.")


if __name__ == "__main__":
    main()
