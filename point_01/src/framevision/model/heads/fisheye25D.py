from typing import Type

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from framevision.camera import fisheye
from framevision.model.components import SoftArgmax2D

DEFAULT_CONV2D_KWARGS = dict(kernel_size=3, padding=1)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

NORMALIZATIONS = {
    "none": nn.Identity,
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
}


class Fisheye25DHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        num_keypoints: int,
        num_views: int,
        conv_hidden_factor: list[int] = [4],
        activation: Type[nn.Module] | str = nn.ReLU,
        normalization: Type[nn.Module] | str = nn.Identity,
        conv2d_kwargs: dict = DEFAULT_CONV2D_KWARGS,
        softargmax_kwargs: dict = {},
    ):
        super().__init__()
        self.num_views = num_views
        self.num_keypoints = num_keypoints
        self.total_keypoints = num_keypoints * num_views

        in_channel = in_channels[-1] * num_views
        feature_channels = [in_channel // factor for factor in conv_hidden_factor]
        out_channel = num_keypoints * 2 * num_views

        channels = [in_channel] + feature_channels + [out_channel]

        activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        normalization = NORMALIZATIONS[normalization] if isinstance(normalization, str) else normalization

        layers = []
        for i in range(len(channels) - 1):
            in_channel = channels[i]
            out_channel = channels[i + 1]
            layers.append(nn.Conv2d(in_channel, out_channel, **conv2d_kwargs))

            if i == len(channels) - 2:
                break

            layers.append(normalization(out_channel))
            layers.append(activation())

        self.head = nn.Sequential(*layers)
        self.softargmax = SoftArgmax2D(num_keypoints * num_views, **softargmax_kwargs)

    def forward(
        self,
        features: list[Float[Tensor, "B T V Ci Hi Wi"]],
        K: Float[Tensor, "B V 3 3"],
        d: Float[Tensor, "B V 4"],
        **kwargs,
    ):
        """Forward pass to predict 3D keypoints from features.

        Args:
            features: list of tensors of shapes (B, T, V, Ci, Hi, Wi)
            K: Camera intrinsic matrix. Shape: (B, V, 3, 3)
            d: Distortion coefficients. Shape: (B, V, 4)

        Returns:
            A dictionary with the following keys:
                - joints_3D_cc: 3D joints coordinates in camera coordinate. Shape: (B, T, V, J, 3)
                - joints_2D_norm: 2D joints coordinates in normalized image coordinate. Shape: (B, T, V, J, 2)
                - depthmaps2D: Depth maps for each keypoint. Shape: (B, T, V, J, H, W)
                - heatmaps2D: 2D heatmaps for each keypoint. Shape: (B, T, V, J, H, W)
        """
        # Flatten the necessary dimensions
        features = self.flatten_TV_dims(features)  # list of tensors of shapes (B * T * V, Ci, Hi, Wi) or (B * T, Ci * V, Hi, Wi)
        feature = features[-1]

        # Forward pass
        feature = self.head(feature)  # Shape: (B * T * V, J * 2, H, W) or (B * T, J * 2 * V, H, W)
        C = self.total_keypoints

        # Generate maps for 2D keypoints and depth
        heatmaps2D = feature[:, 0:C, :, :]  # Shape: (B * T * V, J, H, W) or (B * T, J * V, H, W)
        depthmaps2D = feature[:, C : 2 * C, :, :] * heatmaps2D  # Shape: (B * T * V, J, H, W) or (B * T, J * V, H, W)

        # Compute 2D keypoints indexes and depth values
        xy_coords = self.softargmax(heatmaps2D)  # Shape: (B * T * V, J, 2) or (B * T, J * V, 2)
        depths = depthmaps2D.mean(dim=(-1, -2)).unsqueeze(-1)  # Shape: (B * T * V, J, 1) or (B * T, J * V, 1)

        # Unproject 2D keypoints to 3D
        heatmaps2D, depthmaps2D, xy_coords, depths = self.unflatten_TV_dim(heatmaps2D, depthmaps2D, xy_coords, depths)
        joints_3D_cc = fisheye.unproject(K.unsqueeze(1), d.unsqueeze(1), xy_coords, depths)  # Shape: (B, T, V, J, 3)

        return dict(joints_3D_cc=joints_3D_cc, joints_2D_norm=xy_coords, depthmaps2D=depthmaps2D, heatmaps2D=heatmaps2D)

    def flatten_TV_dims(self, features: list[Tensor]):
        """Flatten the time dimension over the batch dimension and the view dimension over the channel dimension."""
        B, T, V = features[0].shape[:3]
        self._in_shape = (B, T, V)

        chw = [feature.shape[3:] for feature in features]
        return [feature.reshape(B * T, V * C, H, W) for feature, (C, H, W) in zip(features, chw)]

    def unflatten_TV_dim(self, heatmaps2D: Tensor, depthmaps2D: Tensor, xy_coords: Tensor, depths: Tensor):
        """Unflatten the time dimension over the batch dimension and the view dimension over the channel dimension."""
        B, T, V = self._in_shape
        H, W = heatmaps2D.shape[-2:]
        J = self.num_keypoints
        depthmaps2D = depthmaps2D.reshape(B, T, V, J, H, W)
        heatmaps2D = heatmaps2D.reshape(B, T, V, J, H, W)

        xy_coords = xy_coords.reshape(B, T, V, J, 2)
        depths = depths.reshape(B, T, V, J, 1)
        return heatmaps2D, depthmaps2D, xy_coords, depths
