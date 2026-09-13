
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

from step1.data.dataset import FullGraphPredictDataset
from step1.data.transforms import build_full_prediction_graph
from step1.engine.runner import TrainerConfig, run_prediction
from step1.models.spatial_model import SpatialGeneExpressionModel
from torch_geometric.loader import DataLoader as PyGDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict cell-level expression with PhenoMap Step1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a Step1 YAML config",
    )
    return parser.parse_args()


def load_training_artifacts(train_dir: str):
    scaler_path = os.path.join(train_dir, "scale.pkl")
    genes_path = os.path.join(train_dir, "genes.txt")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Expression scaler not found: {scaler_path}")
    if not os.path.exists(genes_path):
        raise FileNotFoundError(f"Gene list not found: {genes_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(genes_path, "r", encoding="utf-8") as f:
        gene_names: List[str] = [line.strip() for line in f if line.strip()]

    print(f"Loaded expression scaler: {scaler_path}")
    print(f"Loaded {len(gene_names)} genes: {genes_path}")
    return scaler, gene_names


def load_h5_and_build_graph(h5_path: str, spot_radius: float):
    print(f"Loading cell features: {h5_path}")
    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = store.keys()
            print(f"  H5 keys: {keys}")

            if "/features" not in keys:
                raise RuntimeError(f"HDF5 file has no '/features' key: {h5_path}")

            features_local_df = store["features"]
            metadata_df = store["metadata"]

            if "/context_features" in keys:
                features_context_df = store["context_features"]
            else:
                print("context_features is absent; using local features as fallback")
                features_context_df = features_local_df.copy()

    except Exception as e:
        raise RuntimeError(f"Failed to load HDF5 features: {e}") from e

    for df in [features_local_df, features_context_df, metadata_df]:
        df.reset_index(inplace=True)
        df.set_index("cell_id", inplace=True)

    common_ids = (
        features_local_df.index
        .intersection(features_context_df.index)
        .intersection(metadata_df.index)
    )
    features_local_df = features_local_df.loc[common_ids]
    features_context_df = features_context_df.loc[common_ids]
    metadata_df = metadata_df.loc[common_ids]
    cell_ids: List[str] = common_ids.tolist()

    print(f"Valid cells: {len(cell_ids)}")

    x_local = torch.tensor(features_local_df.values, dtype=torch.float32)
    x_context = torch.tensor(features_context_df.values, dtype=torch.float32)
    coords = torch.tensor(
        metadata_df[["center_x", "center_y"]].values, dtype=torch.float32
    )

    print(f"Building full-slide graph (radius={spot_radius:.2f}px)...")
    full_graph = build_full_prediction_graph(
        features_local=x_local,
        features_context=x_context,
        coords=coords,
        spot_radius=spot_radius,
    )
    print(f"Graph nodes: {full_graph.num_nodes}; edges: {full_graph.num_edges}")

    return full_graph, cell_ids


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pred_cfg = cfg.get("predict", {})

    for field in ("h5_path", "train_dir", "checkpoint", "output_csv"):
        if not pred_cfg.get(field):
            raise ValueError(f"Required config value predict.{field} is missing")

    if pred_cfg.get("spot_radius") is not None:
        spot_radius = float(pred_cfg["spot_radius"])
        print(f"Using configured graph radius: {spot_radius:.2f}px")
    else:
        diameter_txt = cfg["data"].get("spot_diameter_txt")
        if not diameter_txt or not os.path.exists(diameter_txt):
            raise FileNotFoundError(
                "Set predict.spot_radius or provide data.spot_diameter_txt"
            )
        with open(diameter_txt, "r") as f:
            spot_diameter = float(f.read().strip())
        spot_radius = spot_diameter / 2.0 * cfg["pipeline"]["radius_ratio"]
        print(f"Calculated graph radius: {spot_radius:.2f}px")

    scaler, gene_names = load_training_artifacts(pred_cfg["train_dir"])
    gene_dim = len(gene_names)

    full_graph, cell_ids = load_h5_and_build_graph(pred_cfg["h5_path"], spot_radius)

    dataset = FullGraphPredictDataset(graphs=[full_graph], cell_ids_list=[cell_ids])
    predict_loader = PyGDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_cfg = cfg["model"]
    model = SpatialGeneExpressionModel(
        feat_dim=model_cfg["feat_dim"],
        gene_dim=gene_dim,
        lr=cfg["training"]["lr"],
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
    )

    checkpoint = os.path.join(pred_cfg["train_dir"], pred_cfg["checkpoint"])
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model checkpoint: {checkpoint}")

    train_cfg = cfg["training"]
    trainer_config = TrainerConfig(
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        loss_curve_save_path=None,
    )

    print("Starting inference...")
    result_df: pd.DataFrame = run_prediction(
        model=model,
        predict_loader=predict_loader,
        gene_names=gene_names,
        scaler=scaler,
        config=trainer_config,
    )

    output_csv = pred_cfg["output_csv"]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result_df.to_csv(output_csv)
    print(f"Saved {len(result_df)} cells and {len(result_df.columns)} genes")
    print(f"Predictions: {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    main()
