import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .backbone import Backbone
from .stf import STF


class Frame(nn.Module):
    def __init__(self, backbone: Backbone, stf: STF):
        super().__init__()
        self.backbone = backbone
        self.stf = stf

        T, J, V = self._parse_info(stf)
        joints_cache = torch.zeros((T, V, J, 3), dtype=torch.float32)
        transform_cache = torch.zeros((T, 4, 4), dtype=torch.float32)

        self.register_buffer("_joints_cache", joints_cache)
        self.register_buffer("_transform_cache", transform_cache)

    def forward(
        self,
        images: Float[Tensor, "V C H W"],
        K: Float[Tensor, "V 3 3"],
        d: Float[Tensor, "V 4"],
        left2middle: Float[Tensor, "4 4"],
        right2middle: Float[Tensor, "4 4"],
        middle2world: Float[Tensor, "4 4"],
    ) -> Float[Tensor, "J 3"]:
        # Predict the keypoints of the current frame in the camera coordinate system
        images, K, d = images.unsqueeze(0).unsqueeze(0), K.unsqueeze(0), d.unsqueeze(0)

        joints_cc = self.backbone(images, K, d)["joints_3D_cc"].squeeze()  # Shape: (V, J, 3)

        # Update the cache with the new joints_cc
        self.update_cache(joints_cc, middle2world)

        # Get the joints_cc of the previous frames
        joints_cc_history, m2w_history = self.load_cache()  # Shape: (T, V, J, 3), (T, 4, 4)

        # Apply the spatio-temporal fusion module
        l2m, r2m, m2w, j3D = left2middle.unsqueeze(0), right2middle.unsqueeze(0), m2w_history.unsqueeze(0), joints_cc_history.unsqueeze(0)

        joints = self.stf(j3D, left2middle=l2m, right2middle=r2m, middle2world=m2w)["joints_3D"].squeeze()
        return joints

    def update_cache(self, joints_cc: Float[Tensor, "V J 3"], middle2world: Float[Tensor, "4 4"]):
        self._joints_cache = torch.cat([self._joints_cache[1:], joints_cc.unsqueeze(0)], dim=0)
        self._transform_cache = torch.cat([self._transform_cache[1:], middle2world.unsqueeze(0)], dim=0)

    def load_cache(self):
        return self._joints_cache, self._transform_cache

    def reset(self):
        self._joints_cache.zero_()
        self._transform_cache.zero_()

    def is_warming_up(self) -> bool:
        return torch.any(self._joints_cache == 0)

    def _parse_info(self, stf: STF) -> tuple[int, int, int]:
        # Extract the number of time steps, keypoints, and views from the STF
        time_steps = stf.time_steps if hasattr(stf, "time_steps") else stf._orig_mod.time_steps
        num_keypoints = stf.num_keypoints if hasattr(stf, "num_keypoints") else stf._orig_mod.num_keypoints
        num_views = stf.num_views if hasattr(stf, "num_views") else stf._orig_mod.num_views

        return time_steps, num_keypoints, num_views
