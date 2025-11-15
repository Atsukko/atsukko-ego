# This camera model can work both with normalized and unnormalized K-d / joints2D.
# Throughout the code, we mostly use normalized K-d / joints2D, but nothing stops from using unnormalized ones.
import torch
from jaxtyping import Float
from torch import Tensor


def project(K: Float[Tensor, "... 3 3"], d: Float[Tensor, "... 4"], points_3d: Float[Tensor, "... N 3"]) -> Float[Tensor, "... N 2"]:
    """
    Project 3D points to 2D image coordinates using the fisheye model.

    Args:
        K: Camera intrinsic matrix. Can be normalized (in [0,1]) or unnormalized (in pixel coordinates).
        d: Distortion coefficients.
        points_3d: 3D points to project. Shape (..., N, 3) where N is the number of points.

    Returns:
        Projected 2D pixel coordinates. Shape (..., N, 2). Will be in the same scale as K.
    """
    assert points_3d.shape[-1] == 3, "Last dimension of points_3d should be 3"

    normalized_coords = points_3d[..., :2] / points_3d[..., 2:3]
    distorted_coords = distort(d, normalized_coords)

    # Intrinsic camera matrix parameters
    fx = K[..., 0:1, 0:1]
    fy = K[..., 1:2, 1:2]
    cx = K[..., 0:1, 2:3]
    cy = K[..., 1:2, 2:3]
    skew = K[..., 0:1, 1:2]

    pixels = torch.zeros_like(distorted_coords)
    pixels[..., 0] = (fx * distorted_coords[..., 0:1] + skew * distorted_coords[..., 1:2] + cx).squeeze(-1)
    pixels[..., 1] = (fy * distorted_coords[..., 1:2] + cy).squeeze(-1)

    return pixels


def unproject(
    K: Float[Tensor, "... 3 3"],
    d: Float[Tensor, "... 4"],
    pixels: Float[Tensor, "... N 2"],
    depths: Float[Tensor, "... N 1"],
) -> Float[Tensor, "... N 3"]:
    """
    Unproject 2D pixel coordinates to 3D points using the fisheye model.

    Args:
        K: Camera intrinsic matrix. Can be normalized (in [0,1]) or unnormalized (in pixel coordinates).
        d: Distortion coefficients.
        pixels: 2D pixel coordinates to unproject. Shape (..., N, 2) where N is the number of points. Should be in the same scale as K.
        depths: Depth values for each pixel. Shape (..., N, 1).

    Returns:
        Unprojected 3D points. Shape (..., N, 3).
    """
    assert pixels.shape[-1] == 2, "Last dimension of pixels should be 2"
    assert depths.shape[-1] == 1, "Last dimension of depths should be 1"
    assert pixels.shape[:-1] == depths.shape[:-1], "Shape of pixels and depths must match except for last dimension"

    undistorted_pixels = undistort(K, d, pixels)

    # Intrinsic camera matrix parameters
    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)

    # Normalize undistorted pixel coordinates
    x_prime = (undistorted_pixels[..., 0] - cx) / fx
    y_prime = (undistorted_pixels[..., 1] - cy) / fy

    depth_values = depths.squeeze(-1)
    X = depth_values * x_prime
    Y = depth_values * y_prime
    Z = depth_values

    return torch.stack([X, Y, Z], dim=-1)


def distort(d: Float[Tensor, "... 4"], normalized_coords: Float[Tensor, "... N 2"]) -> Float[Tensor, "... N 2"]:
    """
    Apply fisheye distortion to normalized coordinates.

    Args:
        d: Distortion coefficients.
        normalized_coords: Normalized image coordinates. Shape (..., N, 2) where N is the number of points.


    Returns:
        Distorted coordinates. Shape (..., N, 2). Will be in the same scale as input normalized_coords.
    """
    assert normalized_coords.shape[-1] == 2, "Last dimension of normalized_coords should be 2"

    r = torch.norm(normalized_coords, dim=-1)  # Radial distance from the origin
    theta = torch.atan(r)  # Angle corresponding to the radial distance

    # Distortion coefficients
    k1 = d[..., 0:1]
    k2 = d[..., 1:2]
    k3 = d[..., 2:3]
    k4 = d[..., 3:4]

    r_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)

    scale = r_d / r.clamp(min=1e-8)  # Scaling factor to apply distortion
    distorted_coords = scale.unsqueeze(-1) * normalized_coords  # Apply distortion

    # Handle the case where r is very small to avoid division by zero
    return torch.where(r.unsqueeze(-1) > 1e-8, distorted_coords, normalized_coords)


def undistort(
    K: Float[Tensor, "... 3 3"],
    d: Float[Tensor, "... 4"],
    distorted_pixels: Float[Tensor, "... N 2"],
) -> Float[Tensor, "... N 2"]:
    """
    Undistort 2D pixel coordinates.

    Args:
        K: Camera intrinsic matrix. Can be normalized (in [0,1]) or unnormalized (in pixel coordinates).
        d: Distortion coefficients.
        distorted_pixels: Distorted pixel coordinates. Shape (..., N, 2) where N is the number of points. Should be in the same scale as K.

    Returns:
        Undistorted pixel coordinates. Shape (..., N, 2). Will be in the same scale as input K, d, and distorted_pixels.
    """
    assert distorted_pixels.shape[-1] == 2, "Last dimension of distorted_pixels should be 2"
    assert K.shape[-2:] == (3, 3), "Intrinsic camera matrix should have shape (..., 3, 3)"
    assert d.shape[-1] in (4, 5), "Distortion coefficients should have shape (..., 4) or (..., 5)"
    assert K.shape[0] == d.shape[0], "Batch size of K and d should match"

    # Intrinsic camera matrix parameters
    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)

    # Normalize pixel coordinates
    x_d = (distorted_pixels[..., 0] - cx) / fx
    y_d = (distorted_pixels[..., 1] - cy) / fy

    r_d = torch.sqrt(x_d**2 + y_d**2)  # Radial distance in distorted coordinates

    theta = r_d.clone()  # Initialize theta with r_d

    # Distortion coefficients
    k1 = d[..., 0:1]
    k2 = d[..., 1:2]
    k3 = d[..., 2:3]
    k4 = d[..., 3:4]

    for _ in range(5):  # Iterate to refine theta
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta2 * theta4
        theta8 = theta4**2
        theta = r_d / (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

    scale = torch.tan(theta) / r_d.clamp(min=1e-8)  # Scaling factor to remove distortion
    scale = torch.where(r_d > 1e-8, scale, torch.ones_like(scale))

    x_u = scale * x_d  # Undistorted x coordinates
    y_u = scale * y_d  # Undistorted y coordinates

    # Map back to pixel coordinates
    u_u = fx * x_u + cx
    v_u = fy * y_u + cy

    return torch.stack([u_u, v_u], dim=-1)
