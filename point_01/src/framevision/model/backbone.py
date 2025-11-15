import functools
from typing import Type

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

LazyHead = Type[nn.Module] | functools.partial


class Backbone(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        encoder: nn.Module,
        head_class: LazyHead,
        num_views: int = 2,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.encoder = encoder
        self.keypoint_predictor = head_class(in_channels=encoder.out_channels)
        self.num_views = num_views

    def forward(self, images: Float[Tensor, "B T V 3 H W"], K: Float[Tensor, "B V 3 3"], d: Float[Tensor, "B V 4"], **kwargs):
        """
        Forward pass to predict 3D keypoints from stereo images.

        In the followings, we will use the notation:
            - B: Batch size
            - T: Number of frames (time dimension)
            - V: Number of views
            - H: Height of the image
            - W: Width of the image
            - J: Number of keypoints

        Args:
            images: Shape: (B, T, V, C, H, W). This should have mean 0 and std 1.
            K: Camera intrinsic matrix. Shape: (B, V, 3, 3). This should be normalized for an image of size (1, 1).
            d: Camera distortion coefficients. Shape: (B, V, 4). According to opencv fisheye distortion model.
        """
        # Feature extraction
        features = self.encoder(images)  # Shape: list of (B, T, V, Ci, Hi, Wi)

        # Predict 3D keypoints
        prediction = self.keypoint_predictor(features, K=K, d=d)

        return prediction
