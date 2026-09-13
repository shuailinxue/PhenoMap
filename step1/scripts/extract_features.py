
import argparse
import os
import sys
from pathlib import Path

import yaml

import numpy as np
import pandas as pd
import tifffile as tf
from scipy.ndimage import center_of_mass, find_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1.models.extractors import UNIExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cell-level UNI features from histology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a Step1 YAML config",
    )
    return parser.parse_args()


def load_image(fp: str) -> np.ndarray:
    image = tf.imread(fp)
    if len(image.shape) == 3 and image.shape[0] <= 4:
        image = np.moveaxis(image, 0, -1)  # (C, H, W) → (H, W, C)
    if len(image.shape) == 4:
        image = image[:, :, :, 0]          # (H, W, C, 1) → (H, W, C)
    return image


def extract_cell_features(
    mask_image_path: str,
    histology_image_path: str,
    output_hdf5_path: str,
    local_pad: int,
    context_pad: int,
    batch_size: int,
    device: str | None,
) -> None:
    import torch

    device_obj = torch.device(device) if device else None
    uni = UNIExtractor(batch_size=batch_size, device=device_obj)

    print(f"[1/4] Loading mask: {os.path.basename(mask_image_path)}")
    mask_image = load_image(mask_image_path).astype(np.int32)

    print("[2/4] Locating segmented cells...")
    unique_ids = np.unique(mask_image)
    valid_ids = unique_ids[unique_ids > 0]

    if len(valid_ids) == 0:
        raise ValueError("The segmentation mask contains no cell instances")

    centers_yx = np.array(center_of_mass(mask_image, labels=mask_image, index=valid_ids))
    tgt_pos = np.fliplr(centers_yx).astype(np.float32)

    objs = find_objects(mask_image)
    sizes_1d, final_valid_ids, final_tgt_pos = [], [], []

    for idx, cell_id in enumerate(valid_ids):
        slice_obj = objs[cell_id - 1]
        if slice_obj is not None:
            y_sl, x_sl = slice_obj
            square_size = max(x_sl.stop - x_sl.start, y_sl.stop - y_sl.start)
            sizes_1d.append(square_size)
            final_valid_ids.append(cell_id)
            final_tgt_pos.append(tgt_pos[idx])

    final_valid_ids = np.array(final_valid_ids)
    final_tgt_pos = np.array(final_tgt_pos)
    sizes_1d = np.array(sizes_1d, dtype=np.int32)
    crop_sizes_batch = np.column_stack((sizes_1d, sizes_1d))  # [N, 2]: (w, h)

    print(f"Found {len(final_valid_ids)} valid cells")

    print(f"[3/4] Extracting local features (pad={local_pad})...")
    features_local = uni.extract(
        img_path=histology_image_path,
        spatial_coords=final_tgt_pos,
        crop_sizes=crop_sizes_batch,
        pad=local_pad,
    )

    print(f"[3/4] Extracting context features (pad={context_pad})...")
    features_context = uni.extract(
        img_path=histology_image_path,
        spatial_coords=final_tgt_pos,
        crop_sizes=crop_sizes_batch,
        pad=context_pad,
    )

    print(f"[4/4] Writing features: {output_hdf5_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_hdf5_path)), exist_ok=True)

    metadata_df = pd.DataFrame({
        "cell_id": final_valid_ids,
        "center_x": final_tgt_pos[:, 0],
        "center_y": final_tgt_pos[:, 1],
        "crop_size": sizes_1d,
    }).set_index("cell_id")

    try:
        metadata_df.to_hdf(output_hdf5_path, key="metadata", mode="w")
        with pd.HDFStore(output_hdf5_path, mode="a") as store:
            store.put(
                "features",
                pd.DataFrame(features_local, index=metadata_df.index),
                format="fixed",
            )
            store.put(
                "context_features",
                pd.DataFrame(features_context, index=metadata_df.index),
                format="fixed",
            )
        print(f"Saved {len(metadata_df)} cells with {features_local.shape[1]} features")
    except Exception as e:
        print(f"Failed to save features: {e}")
        raise


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ext = cfg.get("extract", {})

    for field in ("mask_path", "image_path", "output_path"):
        if not ext.get(field):
            raise ValueError(f"Required config value extract.{field} is missing")

    extract_cell_features(
        mask_image_path=ext["mask_path"],
        histology_image_path=ext["image_path"],
        output_hdf5_path=ext["output_path"],
        local_pad=ext.get("local_pad", 10),
        context_pad=ext.get("context_pad", 65),
        batch_size=ext.get("batch_size", 512),
        device=ext.get("device", None),
    )


if __name__ == "__main__":
    main()
