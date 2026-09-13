import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step2.data import load_full_graph, load_target_roi_graph, sample_subgraph
from step2.models import PhenoMapPhenotypeModel, cox_loss_fn, vicreg_loss
from step2.utils.cell_filter import build_cell_mask
from step2.utils.config import apply_overrides, load_config, require
from step2.utils.losses import mmd_loss
from step2.utils.reproducibility import seed_everything
from step2.utils.score_orientation import orient_scores


def parse_args():
    parser = argparse.ArgumentParser(description="PhenoMap Step2 phenotype training")
    parser.add_argument("--config", required=True, help="Path to Step2 YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Override config values: a.b=value")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and imports without loading data")
    return parser.parse_args()


def get_device(device_cfg):
    if device_cfg:
        return torch.device(device_cfg if torch.cuda.is_available() or "cuda" not in device_cfg else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tables(cfg):
    data_cfg = cfg["data"]
    bulk_df = pd.read_csv(data_cfg["bulk_expression"], sep=data_cfg.get("bulk_sep", "\t"), index_col=0)
    clin_df = pd.read_csv(data_cfg["clinical"], sep=data_cfg.get("clinical_sep", "\t"), index_col=0)
    clin_df.index = clin_df.index.astype(str)
    spatial_df = pd.read_csv(data_cfg["spatial_expression"], index_col=0, nrows=5)
    common_genes = bulk_df.index.intersection(spatial_df.columns).tolist()

    phenotype = cfg["phenotype"]
    if phenotype["task"] == "cox":
        clin_df = clin_df.dropna(subset=[phenotype["time_col"], phenotype["event_col"]])
    elif phenotype["task"] == "binary":
        clin_df = clin_df.dropna(subset=[phenotype["label_col"]])
        if phenotype.get("label_map"):
            clin_df[phenotype["label_col"]] = clin_df[phenotype["label_col"]].map(phenotype["label_map"])
        clin_df = clin_df.dropna(subset=[phenotype["label_col"]])
    else:
        raise ValueError("phenotype.task must be 'cox' or 'binary'")

    common_samples = bulk_df.columns.astype(str).intersection(clin_df.index)
    bulk_df.columns = bulk_df.columns.astype(str)
    bulk_df = bulk_df.loc[common_genes, common_samples]
    clin_df = clin_df.loc[common_samples]

    if cfg.get("preprocess", {}).get("bulk_tpm_log1p", False):
        bulk_df = bulk_df.div(bulk_df.sum(axis=0), axis=1) * 1e6
        bulk_df = np.log1p(bulk_df)

    return bulk_df, clin_df, common_genes


def graph_radius(cfg):
    data_cfg = cfg["data"]
    if data_cfg.get("spot_radius") is not None:
        return float(data_cfg["spot_radius"])
    with open(data_cfg["spot_diameter_txt"], "r", encoding="utf-8") as f:
        spot_diameter = float(f.read().strip())
    return (spot_diameter / 2.0) * float(data_cfg.get("spot_radius_ratio", 0.5))


def select_training_indices(h_sc, filter_idx, sample_size):
    if filter_idx is None:
        n = h_sc.shape[0]
        return torch.randperm(n, device=h_sc.device)[: min(sample_size, n)]
    filter_idx = filter_idx.to(h_sc.device)
    if len(filter_idx) == 0:
        raise ValueError("Cell filter selected zero cells")
    return filter_idx[torch.randperm(len(filter_idx), device=h_sc.device)[: min(sample_size, len(filter_idx))]]


def train(cfg):
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg["training"].get("device"))
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk_df, clin_df, common_genes = load_tables(cfg)
    radius = graph_radius(cfg)

    subgraph_size = cfg["training"].get("subgraph_size")
    use_full_graph = subgraph_size is not None

    if use_full_graph:
        print(f"Full-graph mode: loading all cells, subgraph_size={subgraph_size} per epoch")
        full_data = load_full_graph(
            cfg["data"]["spatial_expression"],
            cfg["data"]["mask"],
            common_genes,
            spot_radius=radius,
            seed=cfg.get("seed", 42),
        )
        target_filter_idx = build_cell_mask(
            full_data.cell_ids, cfg["data"].get("annotation_csv"), cfg.get("cell_filter")
        )
        inference_data = full_data
        inference_filter_idx = target_filter_idx
    else:
        roi = cfg["data"].get("roi")
        target_data = load_target_roi_graph(
            cfg["data"]["spatial_expression"],
            cfg["data"]["mask"],
            common_genes,
            roi_limits=roi,
            spot_radius=radius,
            random_downsample=cfg["data"].get("random_downsample", False),
            downsample_frac=cfg["data"].get("downsample_frac", 0.2),
            seed=cfg.get("seed", 42),
        ).to(device)
        full_data = load_target_roi_graph(
            cfg["data"]["spatial_expression"],
            cfg["data"]["mask"],
            common_genes,
            roi_limits=cfg["data"].get("full_roi"),
            spot_radius=radius,
            random_downsample=False,
            seed=cfg.get("seed", 42),
        ).to(device)
        target_filter_idx = build_cell_mask(
            target_data.cell_ids, cfg["data"].get("annotation_csv"), cfg.get("cell_filter")
        )
        inference_data = full_data
        inference_filter_idx = build_cell_mask(
            full_data.cell_ids, cfg["data"].get("annotation_csv"), cfg.get("cell_filter")
        )

    model = PhenoMapPhenotypeModel(
        in_features=len(common_genes),
        align_dim=cfg["model"].get("align_dim", 256),
        gnn_dim=cfg["model"].get("gnn_dim", 16),
        heads=cfg["model"].get("heads", 2),
        task_type=cfg["phenotype"]["task"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr", 1e-3))
    x_bulk = torch.FloatTensor(bulk_df.values).t().to(device)

    phenotype = cfg["phenotype"]
    if phenotype["task"] == "cox":
        times = torch.FloatTensor(clin_df[phenotype["time_col"]].values).to(device)
        events = torch.FloatTensor(clin_df[phenotype["event_col"]].values).to(device)
    else:
        labels_np = clin_df[phenotype["label_col"]].astype(float).values
        labels = torch.FloatTensor(labels_np).unsqueeze(1).to(device)
        num_neg = (labels_np == 0).sum()
        num_pos = (labels_np == 1).sum()
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], device=device)

    losses = cfg["loss"]
    sample_size = cfg["training"].get("sample_size", 20000)
    pbar = tqdm(range(cfg["training"].get("epochs", 200)), unit="epoch")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if use_full_graph:
            sub = sample_subgraph(
                full_data,
                subgraph_size,
                seed=cfg.get("seed", 42) + epoch,
                strategy=cfg["training"].get(
                    "subgraph_strategy", "spatial_block"
                ),
            )
            sub = sub.to(device)
            train_x, train_edge = sub.x, sub.edge_index
            train_filter_idx = None
        else:
            train_x, train_edge = target_data.x, target_data.edge_index
            train_filter_idx = target_filter_idx

        sample_score, z_sc, h_sc, h_bulk, cell_score = model(train_x, train_edge, x_bulk)

        if phenotype["task"] == "cox":
            phenotype_loss = cox_loss_fn(sample_score.flatten(), times, events)
        else:
            phenotype_loss = F.binary_cross_entropy_with_logits(sample_score, labels, pos_weight=pos_weight)

        selected_idx = select_training_indices(h_sc, train_filter_idx, sample_size)
        l_wd = mmd_loss(h_sc[selected_idx], h_bulk)
        l_dec = vicreg_loss(
            z_sc[selected_idx],
            target_std=losses.get("vicreg_target_std", 1.0),
            cov_weight=losses.get("vicreg_cov_weight", 0.1),
            var_weight=losses.get("vicreg_var_weight", 0.1),
        )
        sparse_score = cell_score[train_filter_idx.to(device)] if train_filter_idx is not None else cell_score
        l1_penalty = torch.abs(sparse_score).mean()

        total_loss = (
            phenotype_loss
            + losses.get("mmd_lambda", 1e-4) * l_wd
            + losses.get("vicreg_lambda", 0.01) * l_dec
            + losses.get("l1_lambda", 1e-4) * l1_penalty
        )
        total_loss.backward()
        optimizer.step()

        metrics = {"loss": f"{total_loss.item():.3f}", "mmd": f"{l_wd.item():.3f}"}
        if phenotype["task"] == "cox":
            c_index = concordance_index(times.cpu().detach(), -sample_score.cpu().detach(), events.cpu().detach())
            metrics["C-index"] = f"{c_index:.3f}"
        else:
            probs = torch.sigmoid(sample_score).detach().cpu().numpy()
            true_labels = labels.cpu().numpy()
            try:
                metrics["AUC"] = f"{roc_auc_score(true_labels, probs):.3f}"
                metrics["F1"] = f"{f1_score(true_labels, (probs > 0.5).astype(int)):.3f}"
            except ValueError:
                metrics["AUC"] = "0.500"
                metrics["F1"] = "0.000"
        pbar.set_postfix(metrics)

    model.eval()
    print("\nRunning inference on full graph...")
    inference_chunk_size = cfg["training"].get("inference_chunk_size")
    if use_full_graph or inference_chunk_size:
        chunk_size = inference_chunk_size or 100000
        n_total = inference_data.x.shape[0]
        all_cell_scores = []
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk_idx = torch.arange(start, end)
            chunk_idx_sorted = chunk_idx  # already sorted
            node_mask = torch.zeros(n_total, dtype=torch.bool)
            node_mask[chunk_idx_sorted] = True
            src, dst = inference_data.edge_index
            edge_mask = node_mask[src] & node_mask[dst]
            sub_edge = inference_data.edge_index[:, edge_mask]
            new_id = torch.full((n_total,), -1, dtype=torch.long)
            new_id[chunk_idx_sorted] = torch.arange(len(chunk_idx_sorted))
            sub_edge = new_id[sub_edge]
            chunk_x = inference_data.x[chunk_idx_sorted].to(device)
            chunk_edge = sub_edge.to(device)
            with torch.no_grad():
                chunk_scores = model.predict_cell_scores(chunk_x, chunk_edge)
            all_cell_scores.append(chunk_scores.cpu().numpy().flatten())
            print(f"  Inference chunk {start}-{end} done")
        cell_scores = np.concatenate(all_cell_scores)
    else:
        with torch.no_grad():
            _, _, _, _, full_cell_score = model(
                inference_data.x.to(device), inference_data.edge_index.to(device), x_bulk
            )
            cell_scores = full_cell_score.cpu().numpy().flatten()

    if cfg["output"].get("mask_non_target_to_zero", False) and inference_filter_idx is not None:
        keep = np.zeros_like(cell_scores, dtype=bool)
        keep[inference_filter_idx.numpy()] = True
        cell_scores[~keep] = 0.0

    score_name = cfg["output"].get("score_column", "score")
    raw_scores = pd.Series(cell_scores, index=inference_data.cell_ids, name=score_name)
    orientation = cfg["output"].get("orientation", {})
    if orientation.get("enabled", False):
        raw_file = orientation.get("raw_score_file", f"raw_{cfg['output']['score_file']}")
        raw_scores.to_frame().to_parquet(output_dir / raw_file)
        oriented, diagnostics = orient_scores(
            raw_scores,
            annotation_csv=orientation.get(
                "annotation_csv", cfg["data"].get("annotation_csv")
            ),
            cell_type_col=orientation["cell_type_col"],
            positive_cell_types=orientation["positive_cell_types"],
            statistic=orientation.get("statistic", "median"),
            min_cells=orientation.get("min_cells", 20),
        )
        diagnostics_file = orientation.get(
            "diagnostics_file", "score_orientation.json"
        )
        (output_dir / diagnostics_file).write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        final_scores = oriented
    else:
        final_scores = raw_scores
    final_scores.rename(score_name).to_frame().to_parquet(
        output_dir / cfg["output"]["score_file"]
    )
    torch.save(model.state_dict(), output_dir / cfg["output"].get("checkpoint_file", "model.pt"))


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    if args.dry_run:
        print("Config OK")
        return
    require(cfg, ["data.spatial_expression", "data.mask", "data.bulk_expression", "data.clinical", "output.dir"])
    train(cfg)


if __name__ == "__main__":
    main()
