
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1.data.dataset import SpotGraphDataset
from step1.data.transforms import (
    create_cell_spot_mapping,
    create_radius_graph,
    create_spot_cell_map,
    normalize_gene_expression,
)
from step1.engine.runner import TrainerConfig, run_training
from step1.models.spatial_model import SpatialGeneExpressionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PhenoMap Step1 spatial expression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a Step1 YAML config",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Override YAML values with key.subkey=value entries"
        ),
    )
    return parser.parse_args()


def load_config(config_path: str, overrides: List[str]) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key.subkey=value")
        key_path, value_str = item.split("=", 1)
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                raise ValueError(f"Unknown config override: {key_path}")
            node = node[k]
        try:
            import ast
            node[keys[-1]] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            node[keys[-1]] = value_str

    return cfg


def validate_config(cfg: dict) -> None:
    required_fields = {
        "data.h5_feature_path": cfg["data"]["h5_feature_path"],
        "data.spot_gene_csv": cfg["data"]["spot_gene_csv"],
        "data.spot_coord_csv": cfg["data"]["spot_coord_csv"],
        "data.spot_diameter_txt": cfg["data"]["spot_diameter_txt"],
        "data.output_dir": cfg["data"]["output_dir"],
    }
    missing = [k for k, v in required_fields.items() if v is None]
    if missing:
        raise ValueError(
            "Required config values are missing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines

def build_dataloader(cfg: dict, output_dir: str):
    from torch_geometric.loader import DataLoader as PyGDataLoader

    data_cfg = cfg["data"]
    pipeline_cfg = cfg["pipeline"]
    h5_path = data_cfg["h5_feature_path"]
    gene_csv = data_cfg["spot_gene_csv"]
    coord_csv = data_cfg["spot_coord_csv"]
    diameter_txt = data_cfg["spot_diameter_txt"]

    with open(diameter_txt, "r") as f:
        spot_diameter = float(f.read().strip())
    spot_radius = spot_diameter / 2.0 * pipeline_cfg["radius_ratio"]
    print(f"Spot diameter: {spot_diameter:.2f}px; graph radius: {spot_radius:.2f}px")

    print("Loading cell features...")
    with pd.HDFStore(h5_path, mode="r") as store:
        features_local = store["features"]
        cell_metadata = store["metadata"]
        if "/context_features" in store.keys():
            features_context = store["context_features"]
        else:
            print("context_features is absent; using local features as fallback")
            features_context = features_local.copy()

    for df in [features_local, features_context, cell_metadata]:
        df.reset_index(inplace=True)
        df.set_index("cell_id", inplace=True)

    common_ids = (
        features_local.index
        .intersection(features_context.index)
        .intersection(cell_metadata.index)
    )
    features_local = features_local.loc[common_ids]
    features_context = features_context.loc[common_ids]
    cell_metadata = cell_metadata.loc[common_ids]

    print("Normalizing spot expression...")
    spot_expr_raw = pd.read_csv(gene_csv, index_col=0, header=0)
    gene_list_path = data_cfg.get("gene_list_txt")
    if gene_list_path:
        spot_expr_raw = spot_expr_raw[read_lines(gene_list_path)]
    spot_expr_scaled, scaler = normalize_gene_expression(spot_expr_raw)
    gene_names: List[str] = spot_expr_scaled.columns.tolist()

    os.makedirs(output_dir, exist_ok=True)

    scaler_path = os.path.join(output_dir, "scale.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved expression scaler: {scaler_path}")

    genes_path = os.path.join(output_dir, "genes.txt")
    with open(genes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gene_names) + "\n")
    print(f"Saved {len(gene_names)} gene names: {genes_path}")

    print("Mapping cells to spots...")
    spot_geo_df = pd.read_csv(coord_csv, header=0)
    spot_geo_df.columns = ["spot_id", "spot_center_x", "spot_center_y"]

    cell_data_for_mapping = cell_metadata[["center_x", "center_y"]].reset_index()
    cell_spot_map = create_cell_spot_mapping(
        cell_data_df=cell_data_for_mapping,
        spot_geo_df=spot_geo_df,
        spot_pixel_size=spot_diameter,
    )

    final_spot_ids, X_raw_data, Y_labels = create_spot_cell_map(
        features_local_df=features_local,
        features_context_df=features_context,
        cell_metadata_df=cell_metadata,
        spot_expression_df=spot_expr_scaled,
        cell_spot_map_df=cell_spot_map,
    )
    print(f"Valid spots: {len(final_spot_ids)}")

    print(f"Building radius graphs (radius={spot_radius:.2f}px)...")
    spot_graphs, y_list = [], []

    for spot_id in final_spot_ids:
        raw = X_raw_data[spot_id]
        t_local = torch.tensor(raw["features_local"], dtype=torch.float32)
        t_context = torch.tensor(raw["features_context"], dtype=torch.float32)

        if t_local.shape[0] < 2:
            continue

        graph = create_radius_graph(
            features_local=t_local,
            features_context=t_context,
            cell_coords=raw["coords"],
            radius=spot_radius,
        )
        spot_graphs.append(graph)
        y_list.append(torch.tensor(Y_labels[spot_id], dtype=torch.float32))

    if not spot_graphs:
        raise RuntimeError("No valid spot graph could be constructed")

    y_tensor = torch.stack(y_list, dim=0)
    dataset = SpotGraphDataset(spot_graphs, y_tensor)
    print(f"Training graphs: {len(dataset)}; genes: {dataset.gene_dim}")

    train_loader = PyGDataLoader(
        dataset,
        batch_size=pipeline_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    return train_loader, gene_names


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    validate_config(cfg)

    output_dir = cfg["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    train_loader, gene_names = build_dataloader(cfg, output_dir)
    dataset: SpotGraphDataset = train_loader.dataset

    gene_dim = dataset.gene_dim
    feat_dim = cfg["model"]["feat_dim"]

    print(f"Model dimensions: feat_dim={feat_dim}, gene_dim={gene_dim}")

    model_cfg = cfg["model"]
    model = SpatialGeneExpressionModel(
        feat_dim=feat_dim,
        gene_dim=gene_dim,
        lr=cfg["training"]["lr"],
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
    )

    train_cfg = cfg["training"]
    loss_curve_path = None
    if train_cfg.get("loss_curve_filename"):
        loss_curve_path = os.path.join(output_dir, train_cfg["loss_curve_filename"])

    trainer_config = TrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        log_every_n_steps=train_cfg["log_every_n_steps"],
        loss_curve_save_path=loss_curve_path,
    )

    print("Starting training...")
    trained_model = run_training(model, train_loader, trainer_config)

    ckpt_path = os.path.join(output_dir, "model_final.pt")
    torch.save(trained_model.state_dict(), ckpt_path)
    print(f"Saved model weights: {ckpt_path}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    if loss_curve_path:
        print(f"Saved loss curve: {loss_curve_path}")


if __name__ == "__main__":
    main()
