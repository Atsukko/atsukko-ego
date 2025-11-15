from jaxtyping import Float
from torch import Tensor


def rototranslate(points: Float[Tensor, "... N 3"], transform: Float[Tensor, "... 4 4"]) -> Float[Tensor, "... N 3"]:
    """
    Applies rotation and translation to points using separate components from the transformation matrix.

    Args:
        points: 3D points.
        transform: 4x4 transformation matrix/matrices.
            The leftmost dimensions are considered batch dimensions if present.
            We assume the given transformation is a rigid transformation in SE(3).

    Returns:
        Transformed points. Shape: Same as input points.
    """
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points shape (..., N, 3), got {points.shape}")
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform shape (..., 4, 4), got {transform.shape}")

    # Extract the rotation (upper 3x3 part of the 4x4 matrix)
    rotation = transform[..., :3, :3]  # Shape: (..., 3, 3)

    # Extract the translation (last column, but exclude the bottom-most element)
    translation = transform[..., :3, 3]  # Shape: (..., 3)

    # Apply rotation to points
    rotated_points = points @ rotation.mT  # Shape: (..., N, 3)

    # Apply translation
    translated_points = rotated_points + translation.unsqueeze(-2)  # Shape: (..., N, 3)

    return translated_points
