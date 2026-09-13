import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.checkpoint import checkpoint


def vicreg_loss(z, target_std=1.0, cov_weight=0.1, var_weight=0.1):
    batch_size, num_features = z.shape
    z_centered = z - z.mean(dim=0)
    std_z = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(target_std - std_z))
    cov_z = (z_centered.T @ z_centered) / (batch_size - 1)
    off_diag_mask = ~torch.eye(num_features, dtype=bool, device=z.device)
    cov_loss = cov_z[off_diag_mask].pow(2).sum() / num_features
    return var_weight * var_loss + cov_weight * cov_loss


def sinkhorn_log(C, mu, nu, eps=0.1, max_iter=100):
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    def update(current_u, current_v, cost):
        next_u = eps * (
            torch.log(mu + 1e-8)
            - torch.logsumexp((current_v.unsqueeze(0) - cost) / eps, dim=1)
        )
        next_v = eps * (
            torch.log(nu + 1e-8)
            - torch.logsumexp((next_u.unsqueeze(1) - cost) / eps, dim=0)
        )
        return next_u, next_v

    for _ in range(max_iter):
        if torch.is_grad_enabled() and C.requires_grad:
            u, v = checkpoint(update, u, v, C, use_reentrant=False)
        else:
            u, v = update(u, v, C)
    return torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / eps)


def cox_loss_fn(hazards, times, events):
    _, indices = torch.sort(times, descending=True)
    hazards = hazards[indices]
    events = events[indices]
    exp_hazards = torch.exp(hazards)
    cum_hazards = torch.cumsum(exp_hazards, dim=0)
    log_risk = torch.log(cum_hazards + 1e-8)
    return -torch.sum(events * (hazards - log_risk)) / (torch.sum(events) + 1e-8)


class PhenoMapPhenotypeModel(nn.Module):
    def __init__(self, in_features, align_dim=64, gnn_dim=8, heads=2, task_type="cox"):
        super().__init__()
        self.task_type = task_type
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, align_dim * 2),
            nn.ELU(),
            nn.Linear(align_dim * 2, align_dim),
        )
        if gnn_dim % heads != 0:
            raise ValueError("gnn_dim must be divisible by heads")
        self.conv1 = GATConv(
            in_features, gnn_dim // heads, heads=heads, concat=True
        )
        self.conv2 = GATConv(gnn_dim, gnn_dim, heads=heads, concat=False)
        if task_type == "binary":
            self.classifier = nn.Linear(gnn_dim, 1)
            self.risk_predictor = self.classifier
        else:
            self.risk_predictor = nn.Linear(gnn_dim, 1, bias=False)
            self.classifier = self.risk_predictor

    def forward(self, x_sc, edge_index, x_bulk):
        h_sc = self.shared_encoder(x_sc)
        h_bulk = self.shared_encoder(x_bulk)

        z_sc = self.encode_cells(x_sc, edge_index)

        C = 1 - torch.matmul(F.normalize(h_sc, p=2, dim=1), F.normalize(h_bulk, p=2, dim=1).t())
        mu = torch.full((h_sc.shape[0],), 1.0 / h_sc.shape[0], device=x_sc.device)
        nu = torch.full((x_bulk.shape[0],), 1.0 / x_bulk.shape[0], device=x_sc.device)
        gamma = sinkhorn_log(C, mu, nu)

        sample_repr = torch.matmul(gamma.t(), z_sc)
        sample_score = self.classifier(sample_repr)
        cell_score = self.classifier(z_sc)
        return sample_score, z_sc, h_sc, h_bulk, cell_score

    def encode_cells(self, x_sc, edge_index):
        z_sc = F.elu(self.conv1(x_sc, edge_index))
        return F.elu(self.conv2(z_sc, edge_index))

    def predict_cell_scores(self, x_sc, edge_index):
        return self.classifier(self.encode_cells(x_sc, edge_index))
