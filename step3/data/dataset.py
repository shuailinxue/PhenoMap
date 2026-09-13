import os
import glob
import json
import re
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset
from torchvision import transforms


def _load_he_image(path: str, max_pixels: int = 20000 * 20000) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            level_idx = 0
            for i, lv in enumerate(series.levels):
                h, w = lv.shape[0], lv.shape[1]
                if h * w <= max_pixels:
                    level_idx = i
                    break
            img = series.levels[level_idx].asarray()
            lv = series.levels[level_idx]
            print(f"  Pyramid TIF: level {level_idx} ({lv.shape[0]}x{lv.shape[1]})")
    else:
        img = np.array(Image.open(path).convert('RGB'))

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 4:
        img = img[..., :3]
    return img


class PhenotypeSourceDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.patch_size = config['data']['patch_size']
        self.seq_len = config['model'].get('num_genes', 300)
        self.is_train = is_train
        self.scgpt_model_dir = config['model'].get('scgpt_model_dir')

        self.keep_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.beta_df = pd.read_parquet(config['data']['beta_path'])
        self.he_image = tifffile.imread(config['data']['he_path'])
        mask_image = tifffile.imread(config['data']['mask_path'])

        expr_path = config['data']['sc_expr_path']
        self.expr_df = pd.read_csv(expr_path, index_col=0)

        if self.he_image.ndim == 2:
            self.he_image = np.stack([self.he_image] * 3, axis=-1)
        elif self.he_image.shape[2] >= 4:
            self.he_image = self.he_image[..., :3]

        self.beta_df.index = self.beta_df.index.astype(str)
        coords = center_of_mass(mask_image, labels=mask_image, index=self.beta_df.index.values.astype(int))
        self.beta_df['y'] = np.array(coords)[:, 0]
        self.beta_df['x'] = np.array(coords)[:, 1]

        self.df = self.beta_df.dropna(subset=['x', 'y']).copy()
        self.expr_df.index = self.expr_df.index.astype(str)
        common_cells = self.df.index.intersection(self.expr_df.index)
        self.df = self.df.loc[common_cells]
        self.expr_df = self.expr_df.loc[common_cells]

        print(f"Aligned {len(self.df)} cells across image, coordinates, and expression")

        self.gene_list = self.expr_df.columns.tolist()
        self.n_genes   = len(self.gene_list)
        print(f"Expression genes: {self.n_genes}")
        self._prepare_scgpt_gene_tokens()

        total_samples = len(self.df)

        np.random.seed(config.get('seed', 42))

        shuffled_indices = np.random.permutation(total_samples)

        split_point = int(total_samples * 0.8)

        if self.is_train:
            final_indices = shuffled_indices[:split_point]
        else:
            final_indices = shuffled_indices[split_point:]

        self.df = self.df.iloc[final_indices].copy()
        self.expr_df = self.expr_df.iloc[final_indices].copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cell_id = self.df.index[idx]
        x, y = int(row['x']), int(row['y'])

        half_p = self.patch_size // 2
        y_start, y_end = max(0, y - half_p), min(self.he_image.shape[0], y + half_p)
        x_start, x_end = max(0, x - half_p), min(self.he_image.shape[1], x + half_p)
        patch = self.he_image[y_start:y_end, x_start:x_end]

        pad_y_top = max(0, half_p - y)
        pad_y_bot = max(0, (y + half_p) - self.he_image.shape[0])
        pad_x_left = max(0, half_p - x)
        pad_x_right = max(0, (x + half_p) - self.he_image.shape[1])

        if any([pad_y_top, pad_y_bot, pad_x_left, pad_x_right]):
            patch = np.pad(patch, ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right), (0, 0)), mode='reflect')

        patch_tensor = self.keep_transform(patch)
        hazard_score = torch.tensor(row.iloc[0], dtype=torch.float32)
        coords_tensor = torch.tensor([float(x), float(y)], dtype=torch.float32)

        gene_ids, values = self._build_scgpt_inputs(self.expr_df.loc[cell_id].values.astype(np.float32))

        return {
            'patch': patch_tensor,
            'coords': coords_tensor,
            'gene_ids': torch.tensor(gene_ids, dtype=torch.long),
            'values': torch.tensor(values, dtype=torch.float32),
            'hazard_score': hazard_score
        }

    def _prepare_scgpt_gene_tokens(self):
        if not self.scgpt_model_dir:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("Install huggingface_hub to download scGPT") from exc
            self.scgpt_model_dir = snapshot_download(repo_id="VirtualCell2025/scGPT_human")
        vocab_file = os.path.join(self.scgpt_model_dir, "vocab.json")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"scGPT vocabulary not found: {vocab_file}")

        with open(vocab_file, "r") as f:
            vocab = json.load(f)

        self.pad_token_id = vocab.get("<pad>", vocab.get("<PAD>", 0))
        gene_token_ids = []
        kept_gene_idx = []
        for i, gene in enumerate(self.gene_list):
            token_id = vocab.get(gene)
            if token_id is not None:
                gene_token_ids.append(int(token_id))
                kept_gene_idx.append(i)

        if not gene_token_ids:
            raise ValueError("No expression genes were found in the scGPT vocabulary")

        self.scgpt_gene_ids = np.asarray(gene_token_ids, dtype=np.int64)
        self.scgpt_gene_expr_idx = np.asarray(kept_gene_idx, dtype=np.int64)
        print(f"Genes matched to scGPT vocabulary: {len(self.scgpt_gene_ids)}/{self.n_genes}")

    def _build_scgpt_inputs(self, expr_values):
        expr_values = expr_values[self.scgpt_gene_expr_idx].astype(np.float32)
        gene_ids = self.scgpt_gene_ids

        if len(gene_ids) > self.seq_len:
            keep = np.argsort(expr_values)[-self.seq_len:][::-1]
            gene_ids = gene_ids[keep]
            expr_values = expr_values[keep]

        out_gene_ids = np.full(self.seq_len, self.pad_token_id, dtype=np.int64)
        out_values = np.zeros(self.seq_len, dtype=np.float32)
        n = min(len(gene_ids), self.seq_len)
        out_gene_ids[:n] = gene_ids[:n]
        out_values[:n] = np.log1p(np.clip(expr_values[:n], a_min=0.0, a_max=None))
        return out_gene_ids, out_values


class TargetAnnotationDataset(Dataset):

    def __init__(self, config, section='target_data', he_path_override=None, annotation_path_override=None, slide_id=None):
        target_cfg = config.get(section, {})
        self.target_cfg = target_cfg
        data_cfg = config.get('data', {})
        self.patch_size = target_cfg.get('patch_size', data_cfg.get('patch_size', 224))
        self.stride = target_cfg.get('stride', target_cfg.get('inference_stride', self.patch_size))
        self.tissue_thresh = target_cfg.get('tissue_thresh', data_cfg.get('tissue_thresh', 0.3))

        he_path = he_path_override or target_cfg.get('he_path')
        mask_path = annotation_path_override or target_cfg.get('annotation_path', target_cfg.get('mask_path'))
        if he_path is None or mask_path is None:
            raise ValueError(f"{section}.he_path and {section}.annotation_path are required")
        if he_path == "" or mask_path == "":
            raise ValueError(f"{section}.he_path and {section}.annotation_path cannot be empty")
        self.he_path = he_path
        self.mask_path = mask_path
        self.slide_id = slide_id or os.path.splitext(os.path.basename(he_path))[0]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.he_image = _load_he_image(he_path, target_cfg.get('max_image_pixels', data_cfg.get('max_image_pixels', 200_000_000)))
        annotation = self._load_annotation(mask_path)

        H, W = self.he_image.shape[:2]
        if annotation.shape != (H, W):
            annotation = np.array(
                Image.fromarray(annotation.astype(np.uint8)).resize((W, H), Image.NEAREST)
            ).astype(bool)
        self.annotation = annotation.astype(bool)

        half = self.patch_size // 2
        all_coords = [
            (x, y)
            for y in range(half, H - half + 1, self.stride)
            for x in range(half, W - half + 1, self.stride)
        ]

        self.coords = []
        self.labels = []
        for x, y in all_coords:
            patch = self.he_image[y - half:y + half, x - half:x + half]
            if not self._is_tissue(patch):
                continue
            mask_patch = self.annotation[y - half:y + half, x - half:x + half]
            self.coords.append((x, y))
            self.labels.append(float(mask_patch.mean()))

        max_patches = target_cfg.get('max_patches')
        if max_patches is not None and len(self.coords) > max_patches:
            rng = np.random.default_rng(config.get('seed', 42))
            keep_idx = rng.choice(len(self.coords), size=max_patches, replace=False)
            self.coords = [self.coords[i] for i in keep_idx]
            self.labels = [self.labels[i] for i in keep_idx]

        print(f"Target annotation data: {len(self.coords)}/{len(all_coords)} tissue patches from {os.path.basename(he_path)}")

    def _load_annotation(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            mask = tifffile.imread(path)
        else:
            mask = np.array(Image.open(path))

        if mask.ndim == 3:
            mask = mask[..., 0]

        positive_values = self.target_cfg.get('positive_values')
        if positive_values is not None:
            return np.isin(mask, positive_values)
        return mask > 0

    def _is_tissue(self, patch):
        r = patch[..., 0].astype(np.float32)
        g = patch[..., 1].astype(np.float32)
        b = patch[..., 2].astype(np.float32)
        brightness = (r + g + b) / 3.0
        saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        is_tissue_pixel = (brightness < 230) & (saturation > 15)
        tissue_ratio = is_tissue_pixel.sum() / is_tissue_pixel.size
        return tissue_ratio >= self.tissue_thresh

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        half = self.patch_size // 2
        patch = self.he_image[y - half:y + half, x - half:x + half]
        return {
            'patch': self.transform(patch),
            'coords': torch.tensor([float(x), float(y)], dtype=torch.float32),
            'risk_target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class TargetAnnotationFolderDataset(Dataset):

    def __init__(self, config, section='target_data'):
        target_cfg = config.get(section, {})
        data_cfg = config.get('data', {})
        self.target_cfg = target_cfg
        self.patch_size = target_cfg.get('patch_size', data_cfg.get('patch_size', 224))
        self.stride = target_cfg.get('stride', target_cfg.get('inference_stride', self.patch_size))
        self.tissue_thresh = target_cfg.get('tissue_thresh', data_cfg.get('tissue_thresh', 0.3))
        self.cache_size = target_cfg.get('cache_size', 16)
        self.select_one_slide_per_center = target_cfg.get('select_one_slide_per_center', False)
        self.target_positive_fraction = target_cfg.get('target_positive_fraction', 0.5)
        self.image_cache = OrderedDict()

        he_folder = target_cfg.get('he_folder')
        mask_folder = target_cfg.get('mask_folder')
        if not he_folder or not mask_folder:
            raise ValueError(f"{section}.he_folder and {section}.mask_folder are required")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        image_files = []
        for ext in ('*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg'):
            image_files.extend(glob.glob(os.path.join(he_folder, ext)))
        image_files = sorted(image_files)

        max_images = target_cfg.get('max_images')
        if max_images is not None:
            image_files = image_files[:max_images]

        selected_slides = set(str(s) for s in target_cfg.get('selected_slides', []))
        records = []
        for he_path in image_files:
            name = os.path.basename(he_path)
            logical_slide_id = self._infer_logical_slide_id(name)
            if selected_slides and logical_slide_id not in selected_slides:
                continue

            mask_path = os.path.join(mask_folder, name)
            if not os.path.exists(mask_path):
                continue

            coords, labels = self._scan_slide(config, he_path, mask_path)
            if not coords:
                continue
            center = self._infer_center(logical_slide_id)
            records.append({
                'stem': os.path.splitext(name)[0],
                'logical_slide_id': logical_slide_id,
                'center': center,
                'he_path': he_path,
                'mask_path': mask_path,
                'coords': coords,
                'labels': labels,
            })

        if selected_slides:
            found = sorted({r['logical_slide_id'] for r in records})
            missing = sorted(selected_slides - set(found))
            print(f"Using selected target slides: {found}")
            if missing:
                raise ValueError(f"selected_slides not found in target data: {missing}")
        elif self.select_one_slide_per_center:
            records = self._select_balanced_slides(records)

        self.slides = []
        self.samples = []
        for record in records:
            slide_idx = len(self.slides)
            self.slides.append({
                'stem': record['stem'],
                'logical_slide_id': record['logical_slide_id'],
                'center': record['center'],
                'he_path': record['he_path'],
                'mask_path': record['mask_path'],
            })
            self.samples.extend(
                (slide_idx, x, y, label)
                for (x, y), label in zip(record['coords'], record['labels'])
            )

        max_patches = target_cfg.get('max_patches')
        if max_patches is not None and len(self.samples) > max_patches:
            rng = np.random.default_rng(config.get('seed', 42))
            keep_idx = rng.choice(len(self.samples), size=max_patches, replace=False)
            self.samples = [self.samples[i] for i in keep_idx]

        if not self.samples:
            raise ValueError(f"No target annotation patches found in {he_folder} with masks in {mask_folder}")

        selected_logical = sorted({slide['logical_slide_id'] for slide in self.slides})
        print(
            f"Target annotation folder data: {len(self.samples)} tissue patches "
            f"from {len(self.slides)} image files / {len(selected_logical)} logical slides"
        )

    @staticmethod
    def _infer_logical_slide_id(filename):
        stem = os.path.splitext(filename)[0]
        if "__" in stem:
            stem = stem.split("__", 1)[1]
        stem = re.sub(r"_\[[^\]]+\]$", "", stem)
        return stem

    @staticmethod
    def _infer_center(slide_id):
        if slide_id.startswith("TCGA-"):
            return "TCGA-BRCA"
        if slide_id.startswith("TC_S01_"):
            return "RUMC"
        if re.fullmatch(r"\d+[BS]", slide_id):
            return "JB"
        return "unknown"

    def _select_balanced_slides(self, records):
        if not records:
            return records

        groups = OrderedDict()
        for record in records:
            slide_id = record['logical_slide_id']
            group = groups.setdefault(slide_id, {
                'logical_slide_id': slide_id,
                'center': record['center'],
                'records': [],
                'n_patches': 0,
                'label_sum': 0.0,
            })
            group['records'].append(record)
            group['n_patches'] += len(record['labels'])
            group['label_sum'] += float(np.sum(record['labels']))

        by_center = {}
        for group in groups.values():
            if group['n_patches'] == 0:
                continue
            group['positive_fraction'] = group['label_sum'] / group['n_patches']
            group['balance_error'] = abs(group['positive_fraction'] - self.target_positive_fraction)
            by_center.setdefault(group['center'], []).append(group)

        selected_records = []
        print(
            "Selecting one target slide per center "
            f"(target positive fraction ~= {self.target_positive_fraction:.2f})"
        )
        for center in sorted(by_center):
            candidates = sorted(
                by_center[center],
                key=lambda g: (g['balance_error'], -g['n_patches'], g['logical_slide_id'])
            )
            selected = candidates[0]
            selected_records.extend(selected['records'])
            print(
                f"  {center}: {selected['logical_slide_id']} | "
                f"images={len(selected['records'])}, patches={selected['n_patches']}, "
                f"positive_fraction={selected['positive_fraction']:.3f}"
            )

        return selected_records

    def _scan_slide(self, config, he_path, mask_path):
        he_image = _load_he_image(he_path, self.target_cfg.get('max_image_pixels', 200_000_000))
        annotation = self._load_annotation(mask_path)

        H, W = he_image.shape[:2]
        if annotation.shape != (H, W):
            annotation = np.array(
                Image.fromarray(annotation.astype(np.uint8)).resize((W, H), Image.NEAREST)
            ).astype(bool)

        half = self.patch_size // 2
        all_coords = [
            (x, y)
            for y in range(half, H - half + 1, self.stride)
            for x in range(half, W - half + 1, self.stride)
        ]

        coords = []
        labels = []
        for x, y in all_coords:
            patch = he_image[y - half:y + half, x - half:x + half]
            if not self._is_tissue(patch):
                continue
            mask_patch = annotation[y - half:y + half, x - half:x + half]
            coords.append((x, y))
            labels.append(float(mask_patch.mean()))

        max_patches_per_slide = self.target_cfg.get('max_patches_per_slide')
        if max_patches_per_slide is not None and len(coords) > max_patches_per_slide:
            rng = np.random.default_rng(config.get('seed', 42))
            keep_idx = rng.choice(len(coords), size=max_patches_per_slide, replace=False)
            coords = [coords[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
        return coords, labels

    def _load_annotation(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            mask = tifffile.imread(path)
        else:
            mask = np.array(Image.open(path))

        if mask.ndim == 3:
            mask = mask[..., 0]

        positive_values = self.target_cfg.get('positive_values')
        if positive_values is not None:
            return np.isin(mask, positive_values)
        return mask > 0

    def _is_tissue(self, patch):
        r = patch[..., 0].astype(np.float32)
        g = patch[..., 1].astype(np.float32)
        b = patch[..., 2].astype(np.float32)
        brightness = (r + g + b) / 3.0
        saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        is_tissue_pixel = (brightness < 230) & (saturation > 15)
        tissue_ratio = is_tissue_pixel.sum() / is_tissue_pixel.size
        return tissue_ratio >= self.tissue_thresh

    def _get_image(self, slide_idx):
        if slide_idx in self.image_cache:
            image = self.image_cache.pop(slide_idx)
            self.image_cache[slide_idx] = image
            return image

        image = _load_he_image(self.slides[slide_idx]['he_path'], self.target_cfg.get('max_image_pixels', 200_000_000))
        self.image_cache[slide_idx] = image
        while len(self.image_cache) > self.cache_size:
            self.image_cache.popitem(last=False)
        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slide_idx, x, y, label = self.samples[idx]
        he_image = self._get_image(slide_idx)
        half = self.patch_size // 2
        patch = he_image[y - half:y + half, x - half:x + half]
        return {
            'patch': self.transform(patch),
            'coords': torch.tensor([float(x), float(y)], dtype=torch.float32),
            'risk_target': torch.tensor(label, dtype=torch.float32)
        }


class InferenceDataset(Dataset):

    def __init__(self, config, he_path_override=None, mask_path=None):
        self.patch_size = config['data']['patch_size']
        self.stride = config['data'].get('inference_stride', self.patch_size)
        self.seq_len = config['model'].get('num_genes', 300)
        self.tissue_thresh = config['data'].get('tissue_thresh', 0.7)
        self.filter_background = config['data'].get('filter_background', True)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        he_path = he_path_override or config['data']['he_path']
        self.he_image = _load_he_image(he_path, config['data'].get('max_image_pixels', 200_000_000))

        H, W = self.he_image.shape[:2]
        half = self.patch_size // 2

        if mask_path is not None:
            mask_image = tifffile.imread(mask_path)
            if hasattr(mask_image, 'shape') and mask_image.ndim > 2:
                mask_image = mask_image[..., 0]

            unique_vals = np.unique(mask_image)
            is_binary = len(unique_vals) <= 2  # 0/1 or 0/255 — region mask, not instance mask

            if is_binary:
                tumor_mask = (mask_image > 0)
                if tumor_mask.shape != (H, W):
                    from PIL import Image as _PIL
                    tumor_mask = np.array(
                        _PIL.fromarray(tumor_mask.astype(np.uint8)).resize((W, H), _PIL.NEAREST)
                    ).astype(bool)
                all_coords = self._grid_coords(H, W)
                self.coords = []
                for x, y in all_coords:
                    patch = self._extract_patch(x, y)
                    if tumor_mask[y, x] and self._is_tissue(patch):
                        self.coords.append((x, y))
                if not self.coords:
                    self.coords = all_coords
                    print("Region mask mode fallback: no patch after filtering; using full ROI grid")
                print(f"Region mask mode: {len(self.coords)}/{len(all_coords)} patches inside tumor mask")
            else:
                cell_ids = unique_vals[unique_vals > 0]
                coords_raw = center_of_mass(mask_image, labels=mask_image, index=cell_ids)
                self.coords = [
                    (int(round(cx)), int(round(cy)))
                    for cy, cx in coords_raw
                    if half <= int(round(cy)) < H - half and half <= int(round(cx)) < W - half
                ]
                print(f"Cell mask mode: {len(self.coords)} cells from {os.path.basename(mask_path)}")
        else:
            all_coords = self._grid_coords(H, W)
            self.coords = []
            for x, y in all_coords:
                patch = self._extract_patch(x, y)
                if (not self.filter_background) or self._is_tissue(patch):
                    self.coords.append((x, y))
            if not self.coords:
                self.coords = all_coords
                print("Slide mode fallback: no patch after background filtering; using full ROI grid")
            print(f"Slide mode: {len(self.coords)}/{len(all_coords)} patches kept after background filter")

    def _grid_coords(self, H, W):
        half = self.patch_size // 2
        if H < self.patch_size or W < self.patch_size:
            return [(W // 2, H // 2)]

        xs = list(range(half, W - half + 1, self.stride))
        ys = list(range(half, H - half + 1, self.stride))
        if xs and xs[-1] != W - half:
            xs.append(W - half)
        if ys and ys[-1] != H - half:
            ys.append(H - half)
        if not xs:
            xs = [W // 2]
        if not ys:
            ys = [H // 2]
        return [(x, y) for y in ys for x in xs]

    def _extract_patch(self, x, y):
        half = self.patch_size // 2
        H, W = self.he_image.shape[:2]
        y0, y1 = y - half, y + half
        x0, x1 = x - half, x + half
        src_y0, src_y1 = max(0, y0), min(H, y1)
        src_x0, src_x1 = max(0, x0), min(W, x1)
        patch = self.he_image[src_y0:src_y1, src_x0:src_x1]
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - H)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - W)
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            mode = 'reflect' if patch.shape[0] > 1 and patch.shape[1] > 1 else 'edge'
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode=mode,
            )
        return patch

    def _is_tissue(self, patch):
        r = patch[..., 0].astype(np.float32)
        g = patch[..., 1].astype(np.float32)
        b = patch[..., 2].astype(np.float32)
        brightness = (r + g + b) / 3.0
        saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        is_tissue_pixel = (brightness < 230) & (saturation > 15)
        tissue_ratio = is_tissue_pixel.sum() / is_tissue_pixel.size
        return tissue_ratio >= self.tissue_thresh

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self._extract_patch(x, y)
        patch_tensor = self.transform(patch)
        coords_tensor = torch.tensor([float(x), float(y)], dtype=torch.float32)
        return {
            'patch': patch_tensor,
            'coords': coords_tensor,
        }
