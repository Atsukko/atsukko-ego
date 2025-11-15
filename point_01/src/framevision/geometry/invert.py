import warnings

import torch
from jaxtyping import Float
from torch import Tensor


def invert_SE3(matrix: Float[Tensor, "... 4 4"]) -> Float[Tensor, "... 4 4"]:
    """Invert an SE3 matrix efficiently for matrices of shape (..., 4, 4)."""
    if matrix.dtype != torch.float32:
        warnings.warn(f"Expected matrix of dtype torch.float32, got {matrix.dtype}. Be aware of precision issues.")

    R = matrix[..., :3, :3]
    t = matrix[..., :3, 3]

    # Directly compute the inverse of the rotation
    with torch.autocast(matrix.device.type, enabled=False):
        R_inv = R.mT
        t_inv = -R_inv @ t.unsqueeze(-1)

    # Construct the inverse matrix directly
    inv_matrix = torch.zeros_like(matrix)
    inv_matrix[..., :3, :3] = R_inv
    inv_matrix[..., :3, 3] = t_inv.squeeze(-1)
    inv_matrix[..., 3, 3] = 1  # Set the bottom-right element to 1

    return inv_matrix
