
import torch
import torch.nn as nn


class MultiScaleFusion(nn.Module):

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512) -> None:
        super().__init__()

        self.proj_local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.proj_context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x_local: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        h_l = self.proj_local(x_local)       # [N, hidden_dim]
        h_c = self.proj_context(x_context)   # [N, hidden_dim]

        combined = torch.cat([h_l, h_c], dim=-1)  # [N, hidden_dim * 2]
        z = self.gate_net(combined)

        return z * h_l + (1 - z) * h_c           # [N, hidden_dim]


class SpotPredictorMLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = 512
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ELU(alpha=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
