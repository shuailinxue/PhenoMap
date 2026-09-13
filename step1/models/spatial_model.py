
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add

from .components import MultiScaleFusion, SpotPredictorMLP


class SpatialGeneExpressionModel(pl.LightningModule):

    def __init__(
        self,
        feat_dim: int,
        gene_dim: int,
        lr: float,
        embed_dim: int = 512,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.fusion_module = MultiScaleFusion(input_dim=feat_dim, hidden_dim=embed_dim)

        self.gnn_layers = nn.ModuleList()
        current_dim = embed_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_channels = embed_dim // heads if not is_last else embed_dim
            self.gnn_layers.append(
                GATConv(
                    current_dim,
                    out_channels,
                    heads=heads if not is_last else 1,
                    dropout=dropout,
                    concat=not is_last,
                )
            )
            current_dim = out_channels * heads if not is_last else embed_dim

        self.prediction_head = SpotPredictorMLP(embed_dim, gene_dim)

    def forward(
        self,
        x_local: torch.Tensor,
        x_context: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        H = self.fusion_module(x_local, x_context)

        for i, conv in enumerate(self.gnn_layers):
            H_identity = H
            H = conv(H, edge_index)
            if i < len(self.gnn_layers) - 1:
                H = F.leaky_relu(H)
            H = H + H_identity

        return self.prediction_head(H)

    def _aggregate_and_compute_loss(
        self,
        graph_batch: Batch,
        y_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_local = graph_batch.x
        x_context = graph_batch.x_context

        cell_predictions = self(x_local, x_context, graph_batch.edge_index)

        y_pred = scatter_add(
            src=cell_predictions,
            index=graph_batch.batch,
            dim=0,
            dim_size=y_batch.size(0),
        )

        loss_mse = F.mse_loss(y_pred, y_batch)

        pred_centered = y_pred - torch.mean(y_pred, dim=0, keepdim=True)
        target_centered = y_batch - torch.mean(y_batch, dim=0, keepdim=True)
        covariance = torch.sum(pred_centered * target_centered, dim=0)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=0) + 1e-8)
        target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=0) + 1e-8)
        pcc = covariance / (pred_std * target_std + 1e-8)
        loss_pcc = 1.0 - torch.mean(pcc)

        loss_sparsity = torch.mean(torch.abs(cell_predictions))

        total_loss = loss_mse + 0.1 * loss_pcc + 0.01 * loss_sparsity
        return total_loss, loss_mse, loss_pcc, loss_sparsity

    def training_step(
        self,
        batch: Tuple[Batch, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        graph_batch, y_batch = batch
        total_loss, mse, pcc, sparsity = self._aggregate_and_compute_loss(
            graph_batch, y_batch
        )
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_pcc_loss", pcc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_sparsity", sparsity, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def predict_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> np.ndarray:
        graph_data = batch[0] if isinstance(batch, list) else batch

        x_local = graph_data.x
        x_context = graph_data.x_context if hasattr(graph_data, "x_context") else x_local

        y_pred = self(x_local, x_context, graph_data.edge_index)
        return y_pred.detach().cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
