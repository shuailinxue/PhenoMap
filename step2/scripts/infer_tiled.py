#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import radius_graph
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step2.models import PhenoMapPhenotypeModel
from step2.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--coords", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-size", default=2000, type=float)
    parser.add_argument("--halo", type=float)
    return parser.parse_args()


def get_device(device_cfg: str | None) -> torch.device:
    if device_cfg:
        return torch.device(device_cfg if torch.cuda.is_available() or "cuda" not in device_cfg else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    radius = float(cfg["data"]["spot_radius"])
    halo = args.halo if args.halo is not None else radius * 2
    device = get_device(cfg["training"].get("device"))
    checkpoint = args.checkpoint or str(Path(cfg["output"]["dir"]) / cfg["output"].get("checkpoint_file", "model.pt"))

    expression = pd.read_csv(args.expression, index_col=0)
    expression.index = expression.index.astype(int)
    coords = pd.read_csv(args.coords, usecols=["instance_id", "x", "y"]).set_index("instance_id")
    coords.index = coords.index.astype(int)
    common_ids = expression.index.intersection(coords.index)
    expression = expression.loc[common_ids]
    coords = coords.loc[common_ids]
    features = torch.from_numpy(expression.to_numpy(dtype=np.float32, copy=True))
    xy = coords[["x", "y"]].to_numpy(dtype=np.float32)

    model = PhenoMapPhenotypeModel(
        in_features=features.shape[1],
        align_dim=cfg["model"].get("align_dim", 256),
        gnn_dim=cfg["model"].get("gnn_dim", 16),
        heads=cfg["model"].get("heads", 2),
        task_type=cfg["phenotype"]["task"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    x_edges = np.arange(np.floor(xy[:, 0].min()), np.ceil(xy[:, 0].max()) + args.tile_size, args.tile_size)
    y_edges = np.arange(np.floor(xy[:, 1].min()), np.ceil(xy[:, 1].max()) + args.tile_size, args.tile_size)
    scores = np.full(len(expression), np.nan, dtype=np.float32)
    tile_bounds = [(x0, x0 + args.tile_size, y0, y0 + args.tile_size) for y0 in y_edges[:-1] for x0 in x_edges[:-1]]

    with torch.no_grad():
        for x0, x1, y0, y1 in tqdm(tile_bounds, desc="Inferring tiles"):
            core = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            if not core.any():
                continue
            expanded = (
                (xy[:, 0] >= x0 - halo)
                & (xy[:, 0] < x1 + halo)
                & (xy[:, 1] >= y0 - halo)
                & (xy[:, 1] < y1 + halo)
            )
            expanded_idx = np.flatnonzero(expanded)
            core_idx = np.flatnonzero(core)
            local_lookup = pd.Series(np.arange(len(expanded_idx)), index=expanded_idx)
            local_core_idx = local_lookup.loc[core_idx].to_numpy()
            tile_x = features[expanded_idx].to(device)
            tile_pos = torch.from_numpy(xy[expanded_idx]).to(device)
            edge_index = radius_graph(tile_pos, r=radius, loop=False)
            tile_scores = model.predict_cell_scores(tile_x, edge_index).flatten().cpu().numpy()
            scores[core_idx] = tile_scores[local_core_idx]

    if np.isnan(scores).any():
        raise RuntimeError(f"No score generated for {np.isnan(scores).sum()} cells")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_name = cfg["output"].get("score_column", "score")
    pd.DataFrame({score_name: scores}, index=expression.index).to_parquet(output)
    print(f"Wrote {output}: {len(scores)} cells")


if __name__ == "__main__":
    main()
