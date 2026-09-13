import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        pe_dim = hidden_dim // 4  # sin+cos pairs per coordinate, two coordinates → hidden_dim total
        self.register_buffer('pe_scale', torch.linspace(0, 1, pe_dim))

    def forward(self, coords):
        B, N, _ = coords.shape
        x, y = coords[..., 0], coords[..., 1]

        pe = torch.zeros(B, N, self.hidden_dim, device=coords.device)

        for i in range(self.hidden_dim // 4):
            pe[:, :, 2*i] = torch.sin(x * math.pi * self.pe_scale[i])
            pe[:, :, 2*i + 1] = torch.cos(x * math.pi * self.pe_scale[i])

        for i in range(self.hidden_dim // 4):
            pe[:, :, self.hidden_dim // 2 + 2*i] = torch.sin(y * math.pi * self.pe_scale[i])
            pe[:, :, self.hidden_dim // 2 + 2*i + 1] = torch.cos(y * math.pi * self.pe_scale[i])

        return pe  # shape: [B, N, hidden_dim]


class SpatialFusionModule(nn.Module):

    def __init__(self, vis_dim, spatial_hidden_dim, num_layers=2):
        super().__init__()
        self.vis_dim = vis_dim
        self.spatial_hidden_dim = spatial_hidden_dim

        self.proj_vis = nn.Linear(vis_dim, spatial_hidden_dim)

        fusion_layers = []
        input_dim = spatial_hidden_dim

        for i in range(num_layers):
            fusion_layers.append(nn.LayerNorm(input_dim))
            fusion_layers.append(nn.Linear(input_dim, spatial_hidden_dim))
            if i < num_layers - 1:
                fusion_layers.append(nn.ReLU())
            input_dim = spatial_hidden_dim

        self.fusion_mlp = nn.Sequential(*fusion_layers)

    def forward(self, f_vis, coords):
        proj_vis = self.proj_vis(f_vis)          # [B, N, spatial_hidden_dim]
        f_spatial = self.fusion_mlp(proj_vis)    # [B, N, spatial_hidden_dim]
        return f_spatial


class GeneAlignmentMapper(nn.Module):

    def __init__(self, spatial_dim, gene_dim, num_layers=2):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.gene_dim = gene_dim

        mapper_layers = []
        input_dim = spatial_dim

        for i in range(num_layers):
            mapper_layers.append(nn.LayerNorm(input_dim))
            mapper_layers.append(nn.Linear(input_dim, gene_dim if i == num_layers - 1 else spatial_dim))
            if i < num_layers - 1:
                mapper_layers.append(nn.ReLU())
            input_dim = gene_dim if i == num_layers - 1 else spatial_dim

        self.mapping_mlp = nn.Sequential(*mapper_layers)

    def forward(self, f_spatial):
        e_gene_pred = self.mapping_mlp(f_spatial)  # shape: [B, N, 256] -> [B, N, 512]
        return e_gene_pred


class PhenotypeProjectionHead(nn.Module):

    def __init__(self, gene_dim, text_dim):
        super().__init__()
        self.gene_dim = gene_dim
        self.text_dim = text_dim

        self.projection = nn.Sequential(
            nn.LayerNorm(gene_dim),
            nn.Linear(gene_dim, text_dim)  # shape: [B, N, 512] -> [B, N, 768]
        )

    def forward(self, e_gene_pred):
        z_pred = self.projection(e_gene_pred)  # shape: [B, N, 512] -> [B, N, 768]
        return z_pred


class PhenoMapModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        keep_vis_dim = config['model']['keep_vis_dim']
        keep_text_dim = config['model']['keep_text_dim']
        scgpt_dim = config['model']['scgpt_dim']
        spatial_hidden_dim = config['model']['spatial_hidden_dim']

        self.spatial_fusion = SpatialFusionModule(
            vis_dim=keep_vis_dim,
            spatial_hidden_dim=spatial_hidden_dim,
            num_layers=config['model'].get('spatial_fusion_layers', 2)
        )

        self.gene_alignment = GeneAlignmentMapper(
            spatial_dim=spatial_hidden_dim,
            gene_dim=scgpt_dim,
            num_layers=config['model'].get('alignment_mapper_layers', 2)
        )

        self.phenotype_head = PhenotypeProjectionHead(
            gene_dim=scgpt_dim,
            text_dim=keep_text_dim
        )
        self.disable_text_similarity = config['model'].get('disable_text_similarity', False)
        if self.disable_text_similarity:
            self.risk_classifier = nn.Sequential(
                nn.LayerNorm(scgpt_dim),
                nn.Linear(scgpt_dim, 2),
            )

        self.lambda_align = config['loss']['lambda_align']
        self.lambda_ce = config['loss']['lambda_ce']
        self.scale_factor = config['loss']['scale_factor']
        self.temperature = config['loss']['temperature']

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        f_vis,
        coords,
        e_scgpt,
        v_high,
        v_low,
        hazard_scores=None,
        risk_targets=None,
        label_mode='hazard',
        use_align=True,
    ):

        f_spatial = self.spatial_fusion(f_vis, coords)  # shape: [B, N, 768] -> [B, N, 256]

        e_gene_pred = self.gene_alignment(f_spatial)  # shape: [B, N, 256] -> [B, N, 512]

        z_pred = self.phenotype_head(e_gene_pred)  # shape: [B, N, 512] -> [B, N, 768]

        if use_align and e_scgpt is not None:
            loss_align = self.mse_loss(e_gene_pred, e_scgpt)  # shape: [B, N, 512] vs [B, N, 512] -> scalar
        else:
            loss_align = e_gene_pred.new_tensor(0.0)

        if self.disable_text_similarity:
            logits = self.risk_classifier(e_gene_pred)
        else:
            z_pred_norm = F.normalize(z_pred, p=2, dim=-1)  # shape: [B, N, 768]
            v_high_norm = F.normalize(v_high, p=2, dim=-1)  # shape: [1, 768]
            v_low_norm = F.normalize(v_low, p=2, dim=-1)  # shape: [1, 768]

            sim_high = torch.matmul(z_pred_norm, v_high_norm.T).squeeze(-1) * self.scale_factor
            sim_low = torch.matmul(z_pred_norm, v_low_norm.T).squeeze(-1) * self.scale_factor

            logits = torch.stack([sim_low, sim_high], dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if label_mode == 'hazard':
            if hazard_scores is None:
                raise ValueError("hazard_scores is required when label_mode='hazard'")
            label_high = torch.sigmoid(hazard_scores / self.temperature)
        elif label_mode == 'prob':
            if risk_targets is None:
                raise ValueError("risk_targets is required when label_mode='prob'")
            label_high = risk_targets.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")
        label_low = 1.0 - label_high
        soft_labels = torch.stack([label_low, label_high], dim=-1)

        loss_ce = F.kl_div(log_probs, soft_labels, reduction='batchmean')

        align_weight = self.lambda_align if use_align else 0.0
        loss_total = align_weight * loss_align + self.lambda_ce * loss_ce

        loss_dict = {
            'loss_total': loss_total.item(),
            'loss_align': loss_align.item(),
            'loss_ce': loss_ce.item()
        }

        return logits, loss_total, loss_dict

    def inference(self, f_vis, coords):
        f_spatial = self.spatial_fusion(f_vis, coords)  # shape: [B, N, 768] -> [B, N, 256]
        e_gene_pred = self.gene_alignment(f_spatial)    # shape: [B, N, 256] -> [B, N, 512]
        z_pred = self.phenotype_head(e_gene_pred)       # shape: [B, N, 512] -> [B, N, 768]

        return z_pred

    def predict_logits(self, f_vis, coords, v_high=None, v_low=None):
        f_spatial = self.spatial_fusion(f_vis, coords)
        e_gene_pred = self.gene_alignment(f_spatial)
        z_pred = self.phenotype_head(e_gene_pred)

        if self.disable_text_similarity:
            return self.risk_classifier(e_gene_pred)

        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        v_high_norm = F.normalize(v_high, p=2, dim=-1)
        v_low_norm = F.normalize(v_low, p=2, dim=-1)
        sim_high = torch.matmul(z_pred_norm, v_high_norm.T).squeeze(-1) * self.scale_factor
        sim_low = torch.matmul(z_pred_norm, v_low_norm.T).squeeze(-1) * self.scale_factor
        return torch.stack([sim_low, sim_high], dim=-1)
