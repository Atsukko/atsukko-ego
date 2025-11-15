import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ASTFG(nn.Module):
    """
    Adaptive Spatio-Temporal Fusion Gate Module
    Dynamically adjusts spatial and temporal feature weights based on input content
    """

    def __init__(self, feature_dim, reduction_ratio=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduction_dim = feature_dim // reduction_ratio

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * feature_dim, self.reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduction_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Learnable bias term
        self.bias = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.gate_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spatial_feat, temporal_feat):
        """
        Args:
            spatial_feat: Spatial features from KPA, shape (B, F, N, C)
            temporal_feat: Temporal features from TPA, shape (B, F, N, C)

        Returns:
            fused_feat: Adaptively fused features, shape (B, F, N, C)
        """
        B, F, N, C = spatial_feat.shape

        # Compute global context vectors
        g_s = spatial_feat.mean(dim=2)  # (B, F, C)
        g_t = temporal_feat.mean(dim=2)  # (B, F, C)

        # Concatenate context vectors
        z = torch.cat([g_s, g_t], dim=-1)  # (B, F, 2C)

        # Generate adaptive weights
        w = self.gate_net(z)  # (B, F, 2)
        w_s, w_t = w[..., 0], w[..., 1]  # (B, F)

        # Expand weights for element-wise multiplication
        w_s = w_s.unsqueeze(-1).unsqueeze(-1)  # (B, F, 1, 1)
        w_t = w_t.unsqueeze(-1).unsqueeze(-1)  # (B, F, 1, 1)

        # Adaptive fusion
        fused_feat = w_s * spatial_feat + w_t * temporal_feat + self.bias

        return fused_feat