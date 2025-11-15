import math

import torch
import torch.nn as nn
from torch import Tensor


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        time_steps: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        seq_len, in_dim = time_steps, num_views * num_keypoints * 3

        # Linear embedding layer to project input features to the embedding dimension
        self.embedding = nn.Linear(in_dim, embed_dim)

        # Positional encoding to retain temporal and joint position information
        self.positional_encoding = PositionalEncoding(max_len=seq_len, embed_dim=embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to project back to the original input dimension
        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)

    def forward(self, joints_3D: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            joints_3D: Input tensor of shape (B, T, V, J, 3).

        Returns:
            Output tensor of shape (B, T, J, 3)
        """

        B, T, V, J, _ = joints_3D.shape

        joints_3D_fl_flat = self.flatten(joints_3D)

        # Apply embedding layer
        x = self.embedding(joints_3D_fl_flat)  # Shape: (B, T, embed_dim)

        x = self.positional_encoding(x) + x  # Add positional encoding

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (B, T, embed_dim)

        # Project back to original input dimension for all time steps
        x = self.output_layer(x)  # Shape: (B, T, C)

        return self.unflatten(x)

    def flatten(self, joints_3D: Tensor):
        B, T, V, J, _ = joints_3D.shape
        self._out_shape = (B, T, J, 3)
        return joints_3D.view(B, T, V * J * 3)

    def unflatten(self, joints_3D: Tensor):
        return joints_3D.view(self._out_shape)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int, scale: float = 10000.0, inverted: bool = True):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1) if not inverted else torch.arange(max_len - 1, -1, -1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(scale) / embed_dim))

        pos_enc = torch.zeros(max_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        return self.pos_enc[:, -length:]