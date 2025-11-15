import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class SoftArgmax2D(nn.Module):
    def __init__(self, input_channels: int, beta: float = 10):
        super().__init__()
        beta_tensor = torch.ones(input_channels) * beta
        self.beta = nn.Parameter(beta_tensor.reshape(1, -1, 1))

    def forward(self, heatmaps: Float[Tensor, "B K H W"]) -> Float[Tensor, "B K 2"]:
        B, K, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(B, K, -1)

        # Apply temperature scaling
        scaled_heatmaps = self.beta * heatmaps
        # Subtract max value for numerical stability
        max_values, _ = torch.max(scaled_heatmaps, dim=2, keepdim=True)
        stabilized_heatmaps = scaled_heatmaps - max_values
        # Apply softmax to create probability distribution
        softmax = F.softmax(stabilized_heatmaps, dim=2)

        indices = torch.arange(H * W).unsqueeze(0).unsqueeze(0).to(heatmaps.device)
        y = (indices // W).float() / (H - 1)  # Normalize y to [0, 1]
        x = (indices % W).float() / (W - 1)  # Normalize x to [0, 1]

        x = torch.sum(softmax * x, dim=2)
        y = torch.sum(softmax * y, dim=2)

        return torch.stack((x, y), dim=2)
