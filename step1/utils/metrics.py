
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


@dataclass
class GeneExpressionMetricResult:
    """Gene-level reconstruction metrics and their aligned matrices."""

    metrics: pd.DataFrame
    true_aligned: pd.DataFrame
    pred_aligned: pd.DataFrame
    common_cells: pd.Index
    common_genes: pd.Index


def min_max_normalize_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each gene to [0, 1] for the Step 1 RMSE calculation."""
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    denom = (maxs - mins).replace(0, np.nan)
    return ((df - mins) / denom).fillna(0.0)


def compute_gene_expression_metrics(
    true_expr_path: str | Path,
    pred_expr_path: str | Path,
    output_path: str | Path | None = None,
) -> GeneExpressionMetricResult:
    """Compute per-gene PCC and normalized RMSE on aligned expression tables."""
    true_expr = pd.read_csv(true_expr_path, index_col=0)
    pred_expr = pd.read_csv(pred_expr_path, index_col=0)
    true_expr.index = true_expr.index.astype(str)
    pred_expr.index = pred_expr.index.astype(str)

    common_genes = true_expr.columns.intersection(pred_expr.columns)
    common_cells = true_expr.index.intersection(pred_expr.index)
    true_aligned = true_expr.loc[common_cells, common_genes].copy()
    pred_aligned = pred_expr.loc[common_cells, common_genes].copy()
    true_norm = min_max_normalize_by_gene(true_aligned)
    pred_norm = min_max_normalize_by_gene(pred_aligned)

    rows = []
    for gene in common_genes:
        y_true = true_aligned[gene].to_numpy(dtype=float)
        y_pred = pred_aligned[gene].to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() < 3 or np.std(y_true[valid]) == 0 or np.std(y_pred[valid]) == 0:
            pcc = np.nan
        else:
            pcc = np.corrcoef(y_true[valid], y_pred[valid])[0, 1]

        t_norm = true_norm[gene].to_numpy(dtype=float)
        p_norm = pred_norm[gene].to_numpy(dtype=float)
        valid_norm = np.isfinite(t_norm) & np.isfinite(p_norm)
        rmse = np.sqrt(np.mean((t_norm[valid_norm] - p_norm[valid_norm]) ** 2))
        rows.append({"gene": gene, "PCC": pcc, "RMSE": rmse})

    metrics = pd.DataFrame(rows).dropna()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(output_path, index=False)

    return GeneExpressionMetricResult(
        metrics=metrics,
        true_aligned=true_aligned,
        pred_aligned=pred_aligned,
        common_cells=common_cells,
        common_genes=common_genes,
    )


class LossCurveCallback(Callback):

    def __init__(self, save_path: str = "loss_curve.png") -> None:
        super().__init__()
        self.save_path = save_path
        self.train_losses: List[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            loss = metrics["train_loss"].detach().cpu().item()
            self.train_losses.append(loss)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        parent_dir = os.path.dirname(self.save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        plt.figure()
        plt.plot(self.train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=300)
        plt.close()
        print(f"Saved loss curve: {os.path.abspath(self.save_path)}")
