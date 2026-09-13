
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
from huggingface_hub import login
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))


class ROIDataset(Dataset):

    def __init__(self, img_list: List[np.ndarray], transform) -> None:
        self.images_lst = img_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_lst)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pil_image = Image.fromarray(self.images_lst[idx].astype("uint8"))
        return self.transform(pil_image)


class UNIExtractor:

    def __init__(
        self,
        batch_size: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        token = os.environ.get("HF_TOKEN")
        if token:
            try:
                login(token)
            except Exception as exc:
                print(f"Hugging Face login failed; trying the local cache: {exc}")
        else:
            print("HF_TOKEN is not set; trying the local model cache")

        try:
            self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        except Exception as exc:
            print(f"Failed to load UNI from Hugging Face; trying the local cache: {exc}")
            self.model = self._load_uni_from_local_cache()
        self.model.eval().to(self.device)

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def _load_uni_from_local_cache(self) -> torch.nn.Module:
        cache_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--MahmoodLab--uni"
            / "snapshots"
        )
        candidates = sorted(cache_root.glob("*/pytorch_model.bin"))
        if not candidates:
            raise FileNotFoundError(
                "MahmoodLab/uni was not found in the Hugging Face cache. "
                "Set HF_TOKEN and rerun feature extraction."
            )
        weight_path = candidates[-1]
        config_path = weight_path.parent / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = timm.create_model(
            cfg.get("architecture", "vit_large_patch16_224"),
            pretrained=False,
            num_classes=cfg.get("num_classes", 0),
            init_values=cfg.get("init_values", 1.0),
            dynamic_img_size=cfg.get("dynamic_img_size", True),
        )
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.pretrained_cfg = cfg.get("pretrained_cfg", model.pretrained_cfg)
        return model

    def crop_image(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        crop_size: Tuple[int, int],
        pad: int,
    ) -> np.ndarray:
        y_max, x_max = img.shape[:2]
        w, h = crop_size

        x_start = x - int(w // 2) - pad
        y_start = y - int(h // 2) - pad
        x_end = x + int(w // 2) + pad
        y_end = y + int(h // 2) + pad

        left = max(0, x_start)
        top = max(0, y_start)
        right = min(x_max, x_end)
        bottom = min(y_max, y_end)

        return img[top:bottom, left:right]

    def extract(
        self,
        img_path: str,
        spatial_coords: np.ndarray,
        crop_sizes: np.ndarray,
        pad: int,
    ) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Cannot read histology image: {img_path}"
            )
        img = np.array(img)

        feature_embs: List[np.ndarray] = []
        chunk_size = max(self.batch_size * 8, self.batch_size)
        total = len(spatial_coords)

        with torch.inference_mode():
            for start in range(0, total, chunk_size):
                end = min(total, start + chunk_size)
                sub_images = [
                    self.crop_image(img, int(x), int(y), crop_size, pad)
                    for (x, y), crop_size in zip(
                        spatial_coords[start:end], crop_sizes[start:end]
                    )
                ]
                dataset = ROIDataset(sub_images, self.transform)
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                for batch in loader:
                    batch = batch.to(self.device)
                    emb = self.model(batch)
                    feature_embs.append(emb.cpu().numpy())

        return np.concatenate(feature_embs, axis=0)
