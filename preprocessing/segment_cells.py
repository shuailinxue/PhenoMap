#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from cellpose import models
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment nuclei/cells from a large HE image using Cellpose tiles."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", default="nuclei")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--flow-threshold", type=float, default=0.8)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--invert", action="store_true", default=True)
    parser.add_argument("--no-invert", dest="invert", action="store_false")
    parser.add_argument("--blank-gray-threshold", type=float, default=245.0)
    parser.add_argument("--blank-saturation-threshold", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Optional smoke-test limit. Does not write the final mask unless --output is used with --overwrite.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def iter_tiles(height: int, width: int, tile_size: int, overlap: int):
    stride = tile_size - 2 * overlap
    if stride <= 0:
        raise ValueError("--tile-size must be larger than 2 * --overlap")
    n_y = max(1, math.ceil(height / stride))
    n_x = max(1, math.ceil(width / stride))
    for iy in range(n_y):
        y0_inner = iy * stride
        y1_inner = min(height, y0_inner + stride)
        if iy == n_y - 1:
            y1_inner = height
        y0 = max(0, y0_inner - overlap)
        y1 = min(height, y1_inner + overlap)
        for ix in range(n_x):
            x0_inner = ix * stride
            x1_inner = min(width, x0_inner + stride)
            if ix == n_x - 1:
                x1_inner = width
            x0 = max(0, x0_inner - overlap)
            x1 = min(width, x1_inner + overlap)
            yield x0, x1, y0, y1, x0_inner, x1_inner, y0_inner, y1_inner


def is_blank_tile(tile_rgb: np.ndarray, gray_threshold: float, saturation_threshold: float) -> bool:
    if tile_rgb.size == 0:
        return True
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    return float(gray.mean()) > gray_threshold and float(hsv[:, :, 1].mean()) < saturation_threshold


def paste_inner_mask(
    out_mask: np.ndarray,
    tile_mask: np.ndarray,
    tile_bounds: tuple[int, int, int, int, int, int, int, int],
    next_id: int,
) -> int:
    x0, x1, y0, y1, x0_inner, x1_inner, y0_inner, y1_inner = tile_bounds
    src_x0 = x0_inner - x0
    src_x1 = src_x0 + (x1_inner - x0_inner)
    src_y0 = y0_inner - y0
    src_y1 = src_y0 + (y1_inner - y0_inner)
    inner = tile_mask[src_y0:src_y1, src_x0:src_x1]

    labels = np.unique(inner)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return next_id

    pasted = np.zeros(inner.shape, dtype=np.uint32)
    for old_id in labels:
        pasted[inner == old_id] = next_id
        next_id += 1
    out_mask[y0_inner:y1_inner, x0_inner:x1_inner] = pasted
    return next_id


def segment_image(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        print(f"Output exists, skipping: {args.output}")
        return

    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    height, width = image.shape[:2]
    print(f"Image shape: {height} x {width} x {image.shape[2]}")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
        gpu = True
    else:
        device = torch.device("cpu")
        gpu = False
    print(f"Using device: {device}")

    model = models.CellposeModel(gpu=gpu, model_type=args.model_type, device=device)
    out_mask = np.zeros((height, width), dtype=np.uint32)
    next_id = 1
    skipped = 0
    total_tiles = sum(1 for _ in iter_tiles(height, width, args.tile_size, args.overlap))

    for tile_no, bounds in enumerate(
        tqdm(iter_tiles(height, width, args.tile_size, args.overlap), total=total_tiles, desc="Cellpose tiles"),
        start=1,
    ):
        if args.max_tiles is not None and tile_no > args.max_tiles:
            break
        x0, x1, y0, y1, *_ = bounds
        tile = image[y0:y1, x0:x1]
        if is_blank_tile(tile, args.blank_gray_threshold, args.blank_saturation_threshold):
            skipped += 1
            continue

        masks, _, _ = model.eval(
            tile,
            channels=[args.channel, 0],
            diameter=args.diameter,
            min_size=args.min_size,
            invert=args.invert,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        next_id = paste_inner_mask(out_mask, masks.astype(np.uint32, copy=False), bounds, next_id)

    if args.max_tiles is not None:
        print(f"Stopped after smoke-test max tiles: {args.max_tiles}")
    print(f"Skipped blank tiles: {skipped}")
    print(f"Detected instances pasted: {next_id - 1}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing mask: {args.output}")
    tifffile.imwrite(args.output, out_mask, bigtiff=True, photometric="minisblack")
    print("Done.")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    segment_image(args)


if __name__ == "__main__":
    main()
