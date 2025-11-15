import torch
from torch import Tensor

from .invert import invert_SE3


def compute_relpose_to_floor(pose: Tensor, align_z_to: str = "x", y_offset: float = -0.75, randomize: bool = False) -> Tensor:
    """
    Compute the matrix necessary to project of a given pose to the floor.
    With y-axis aligned with the global y-axis and z-axis on the floor.

    Args:
        pose: The input pose to project. Shape: (..., 4, 4)
        align_z_to: The axis of the original pose to align the new z-axis to, ib the global xz plane.
        y_offset: The offset to apply to the y-axis.
        randomize: If True, randomize the xz plane rotation.

    Returns:
        relpose: The relative pose to transform the input pose to the floor projection. Shape: (..., 4, 4)
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[align_z_to]
    # Step 1: Compute the new z axis by projecting it in the global xz plane
    if not randomize:
        axis = pose[..., :3, axis_idx].clone()
        axis[..., 1] = 0
    else:
        axis = torch.randn_like(pose[..., :3, axis_idx])
        axis[..., 1] = 0

    projection_norm = torch.norm(axis, dim=-1, keepdim=True)
    new_z_axis = axis / projection_norm

    # Step 2: Compute the new y axis to be aligned with the global y axis
    new_y_axis = torch.tensor([0, 1, 0], device=new_z_axis.device, dtype=new_z_axis.dtype).expand_as(axis)

    # Step 3: Compute the new x axis by crossing the new y and z axes
    new_x_axis = torch.cross(new_y_axis, new_z_axis, dim=-1)

    # Step 4: Compute the new translation by projecting to the floor
    translation = pose[..., :3, 3].clone()
    translation[..., 1] = 0

    # Create the new pose
    T = torch.zeros_like(pose)
    T[..., :3, 0] = new_x_axis
    T[..., :3, 1] = new_y_axis
    T[..., :3, 2] = new_z_axis
    T[..., :3, 3] = translation
    T[..., 3, 3] = 1

    # Compute the relative pose between the two poses
    relpose = invert_SE3(T) @ pose

    # Apply the y offset
    relpose[..., 1, 3] += y_offset

    return relpose


def compute_relpose_to_center(pose: Tensor, y_offset: float = -0.75) -> Tensor:
    """
    Compute the matrix necessary to project of a given pose to the same orientation but centered at the origin.

    Args:
        pose: The input pose to project. Shape: (..., 4, 4)
    """
    relpose = pose.clone()
    relpose[..., 0, 3] = 0
    relpose[..., 1, 3] += y_offset
    relpose[..., 2, 3] = 0

    return relpose
