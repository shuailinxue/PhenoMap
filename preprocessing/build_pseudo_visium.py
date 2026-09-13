#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pseudo-Visium spot counts for PhenoMap Step1"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--he", default="he.jpg")
    parser.add_argument("--pixel-size", default="pixel-size-raw.txt")
    parser.add_argument("--cell-locs", default="cell_locs.tsv")
    parser.add_argument("--cell-gene-matrix", default="cell_gene_matrix.tsv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix before output extensions, e.g. '_filtered'.",
    )
    parser.add_argument("--spot-diameter-um", type=float, default=55.0)
    parser.add_argument("--spot-spacing-um", type=float, default=100.0)
    parser.add_argument("--mask-downsample", type=int, default=32)
    parser.add_argument("--tissue-gray-threshold", type=int, default=238)
    parser.add_argument("--tissue-saturation-threshold", type=int, default=8)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.25)
    parser.add_argument("--min-cells-per-spot", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--max-spots",
        type=int,
        help="Optional smoke-test limit after tissue filtering.",
    )
    return parser.parse_args()


def read_pixel_size(path: Path) -> float:
    return float(path.read_text().strip())


def load_he_shape_and_tissue_mask(
    he_path: Path,
    downsample: int,
    gray_threshold: int,
    saturation_threshold: int,
) -> tuple[tuple[int, int], np.ndarray]:
    image_bgr = cv2.imread(str(he_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read HE image: {he_path}")

    height, width = image_bgr.shape[:2]
    small = cv2.resize(
        image_bgr,
        (max(1, width // downsample), max(1, height // downsample)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    tissue = (gray < gray_threshold) & (hsv[:, :, 1] > saturation_threshold)
    tissue = tissue.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel, iterations=1)
    return (height, width), tissue.astype(bool)


def disk_offsets(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = xx * xx + yy * yy <= radius * radius
    return np.column_stack([xx[keep], yy[keep]])


def tissue_fraction(
    tissue_mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius_px: float,
    downsample: int,
    offsets_cache: dict[int, np.ndarray],
) -> float:
    radius_mask = max(1, int(round(radius_px / downsample)))
    if radius_mask not in offsets_cache:
        offsets_cache[radius_mask] = disk_offsets(radius_mask)
    offsets = offsets_cache[radius_mask]
    cx = int(round(center_x / downsample))
    cy = int(round(center_y / downsample))
    xs = cx + offsets[:, 0]
    ys = cy + offsets[:, 1]
    valid = (
        (xs >= 0)
        & (xs < tissue_mask.shape[1])
        & (ys >= 0)
        & (ys < tissue_mask.shape[0])
    )
    if not np.any(valid):
        return 0.0
    return float(tissue_mask[ys[valid], xs[valid]].mean())


def generate_hex_centers(
    image_shape: tuple[int, int],
    spacing_px: float,
    radius_px: float,
) -> pd.DataFrame:
    height, width = image_shape
    row_step = spacing_px * np.sqrt(3.0) / 2.0
    centers: list[tuple[float, float]] = []
    y = radius_px
    row = 0
    while y <= height - radius_px:
        x_offset = radius_px + (spacing_px / 2.0 if row % 2 else 0.0)
        x = x_offset
        while x <= width - radius_px:
            centers.append((x, y))
            x += spacing_px
        y += row_step
        row += 1
    return pd.DataFrame(centers, columns=["x", "y"])


def build_spot_cell_mapping(
    locs: pd.DataFrame,
    spots: pd.DataFrame,
    radius_px: float,
    min_cells_per_spot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = cKDTree(locs[["x", "y"]].to_numpy(dtype=np.float32))
    spot_centers = spots[["x", "y"]].to_numpy(dtype=np.float32)
    hit_lists = tree.query_ball_point(spot_centers, r=radius_px)

    kept_spot_rows: list[int] = []
    mapping_frames: list[pd.DataFrame] = []
    cell_ids = locs["spot"].to_numpy()
    for spot_idx, hit in enumerate(hit_lists):
        if len(hit) < min_cells_per_spot:
            continue
        kept_spot_rows.append(spot_idx)
        mapping_frames.append(
            pd.DataFrame(
                {
                    "spot": cell_ids[np.asarray(hit, dtype=np.int64)],
                    "pseudo_spot": len(kept_spot_rows) - 1,
                }
            )
        )

    if not mapping_frames:
        raise RuntimeError("No pseudo spots contain enough cells.")

    kept = spots.iloc[kept_spot_rows].copy().reset_index(drop=True)
    kept.insert(0, "spots", np.arange(len(kept), dtype=np.int64))
    mapping = pd.concat(mapping_frames, ignore_index=True)
    return kept, mapping


def aggregate_expression(
    matrix_path: Path,
    mapping: pd.DataFrame,
    num_spots: int,
    chunksize: int,
) -> tuple[pd.DataFrame, list[str]]:
    sep = "," if matrix_path.suffix.lower() == ".csv" else "\t"
    header = pd.read_csv(matrix_path, sep=sep, nrows=0)
    if "spot" not in header.columns:
        raise ValueError(f"{matrix_path} must contain a first-column cell id named 'spot'")
    genes = [col for col in header.columns if col != "spot"]
    counts = np.zeros((num_spots, len(genes)), dtype=np.float32)

    mapping = mapping.copy()
    mapping["spot"] = mapping["spot"].astype(str)
    usecols = ["spot", *genes]
    reader = pd.read_csv(matrix_path, sep=sep, usecols=usecols, chunksize=chunksize)
    for chunk in tqdm(reader, desc="Aggregating expression chunks"):
        chunk["spot"] = chunk["spot"].astype(str)
        merged = mapping.merge(chunk, on="spot", how="inner")
        if merged.empty:
            continue
        grouped = merged.groupby("pseudo_spot", sort=False)[genes].sum()
        counts[grouped.index.to_numpy(dtype=np.int64), :] += grouped.to_numpy(dtype=np.float32)

    out = pd.DataFrame(counts, columns=genes)
    out.insert(0, "spots", np.arange(num_spots, dtype=np.int64))
    return out, genes


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size = read_pixel_size(data_dir / args.pixel_size)
    spot_diameter_px = args.spot_diameter_um / pixel_size
    spot_radius_px = spot_diameter_px / 2.0
    spot_spacing_px = args.spot_spacing_um / pixel_size

    image_shape, tissue_mask = load_he_shape_and_tissue_mask(
        data_dir / args.he,
        args.mask_downsample,
        args.tissue_gray_threshold,
        args.tissue_saturation_threshold,
    )
    suffix = args.output_suffix
    cv2.imwrite(
        str(output_dir / f"tissue_mask_downsampled{suffix}.png"),
        tissue_mask.astype(np.uint8) * 255,
    )

    candidates = generate_hex_centers(image_shape, spot_spacing_px, spot_radius_px)
    offsets_cache: dict[int, np.ndarray] = {}
    fractions = [
        tissue_fraction(
            tissue_mask,
            row.x,
            row.y,
            spot_radius_px,
            args.mask_downsample,
            offsets_cache,
        )
        for row in tqdm(candidates.itertuples(index=False), total=len(candidates), desc="Filtering tissue spots")
    ]
    candidates["tissue_fraction"] = fractions
    candidates = candidates[candidates["tissue_fraction"] >= args.min_tissue_fraction]
    if args.max_spots is not None:
        candidates = candidates.head(args.max_spots).copy()
    if candidates.empty:
        raise RuntimeError("No pseudo spots passed the tissue-mask filter.")

    locs = pd.read_csv(data_dir / args.cell_locs, sep="\t", usecols=["spot", "x", "y"])
    kept_locs, mapping = build_spot_cell_mapping(
        locs=locs,
        spots=candidates[["x", "y"]],
        radius_px=spot_radius_px,
        min_cells_per_spot=args.min_cells_per_spot,
    )
    pseudo_counts, genes = aggregate_expression(
        matrix_path=data_dir / args.cell_gene_matrix,
        mapping=mapping,
        num_spots=len(kept_locs),
        chunksize=args.chunksize,
    )

    pseudo_counts.to_csv(output_dir / f"PseudoVisium{suffix}.csv", index=False)
    kept_locs[["spots", "x", "y"]].to_csv(
        output_dir / f"locs{suffix}.csv", index=False, float_format="%.2f"
    )
    (output_dir / f"genes{suffix}.txt").write_text("\n".join(genes) + "\n")
    (output_dir / f"spot_diameter_pixel{suffix}.txt").write_text(f"{spot_diameter_px:.6f}\n")

    manifest = {
        "data_dir": str(data_dir),
        "image_shape_hw": list(image_shape),
        "pixel_size_um_per_px": pixel_size,
        "spot_diameter_um": args.spot_diameter_um,
        "spot_spacing_um": args.spot_spacing_um,
        "spot_diameter_px": spot_diameter_px,
        "spot_spacing_px": spot_spacing_px,
        "candidate_spots": int(len(fractions)),
        "tissue_filtered_spots": int(len(candidates)),
        "final_spots": int(len(kept_locs)),
        "cell_spot_assignments": int(len(mapping)),
        "genes": int(len(genes)),
        "min_tissue_fraction": args.min_tissue_fraction,
        "min_cells_per_spot": args.min_cells_per_spot,
    }
    with (output_dir / f"pseudo_visium_manifest{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
