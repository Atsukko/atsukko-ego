from typing import Optional

import timm
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        out_indices: Optional[list[int]] = None,
        num_views: int = 2,
        **kwargs,
    ):
        """Feature extractor based on timm models.

        Args:
            name: Name of the model to use.
            pretrained: Whether to use a pretrained model.
            out_indices: Indices of the layers to extract features from.
            num_views: Number of views to process.

        Common Kwargs:
            output_stride: Output stride of the model, default is 32.
        """
        super().__init__()

        in_channels = 3
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
            **kwargs,
        )
        self.num_views = num_views

    def forward(self, images: Float[Tensor, "B T V 3 H W"]) -> list[Tensor]:
        """Forward pass to extract features from images.

        Args:
            images: Input images. Shape: (B, T, V, C, H, W)
        """
        images = self.flatten(images)
        features = self.model(images)
        return self.unflatten(features)

    def flatten(self, images: Tensor, strategy: str = None) -> Tensor:
        B, T, V, C, H, W = images.shape
        self._inshape = (B, T, V)
        return images.reshape(B * T * V, C, H, W)

    def unflatten(self, features: list[Tensor]) -> list[Tensor]:
        B, T, V = self._inshape

        unflatten_features = []
        for feature in features:
            _, C, H, W = feature.shape

            unflat_feature = feature.reshape(B, T, V, C, H, W)
            unflatten_features.append(unflat_feature)

        return unflatten_features

    @property
    def out_channels(self) -> list[int]:
        return self.model.feature_info.channels()

    @property
    def out_strides(self) -> list[int]:
        return self.model.feature_info.reduction()
