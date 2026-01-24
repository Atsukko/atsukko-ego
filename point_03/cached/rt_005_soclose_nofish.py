from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torchvision
import typer
import lightning as L
from tqdm import tqdm
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

import framevision
from framevision.model import Frame
from framevision.pl_wrappers import FrameDataModule

app = typer.Typer()

# ✅ 修正后的骨架连接
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13), (13, 14)
]


def distort(d: Float[Tensor, "... 4"], normalized_coords: Float[Tensor, "... N 2"]) -> Float[Tensor, "... N 2"]:
    assert normalized_coords.shape[-1] == 2, "Last dimension of normalized_coords should be 2"
    r = torch.norm(normalized_coords, dim=-1)
    theta = torch.atan(r)

    k1 = d[..., 0:1]
    k2 = d[..., 1:2]
    k3 = d[..., 2:3]
    k4 = d[..., 3:4]

    r_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)
    scale = r_d / r.clamp(min=1e-8)
    distorted_coords = scale.unsqueeze(-1) * normalized_coords

    return torch.where(r.unsqueeze(-1) > 1e-8, distorted_coords, normalized_coords)


def project_3d_to_2d(points_3d, K, T_world_to_cam, width=256, height=256):
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_cam = (T_world_to_cam @ points_3d_h.T).T
    z = points_cam[:, 2]
    eps = 1e-6
    projected = (K @ points_cam[:, :3].T).T
    u = projected[:, 0] / (z + eps)
    v = projected[:, 1] / (z + eps)
    mask = z <= 0
    u[mask] = np.nan
    v[mask] = np.nan
    return np.stack([u, v], axis=1)


def undistort_images(images, K, d):
    B, C, H, W = images.shape
    device = images.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_grid, y_grid], dim=-1).expand(B, -1, -1, -1)

    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)

    normalized_coords = torch.zeros_like(pixel_coords)
    normalized_coords[..., 0] = (pixel_coords[..., 0] - cx) / fx
    normalized_coords[..., 1] = (pixel_coords[..., 1] - cy) / fy

    flat_coords = normalized_coords.view(B, -1, 2)
    distorted_flat = distort(d, flat_coords)
    distorted_coords = distorted_flat.view(B, H, W, 2)

    u_src = distorted_coords[..., 0] * fx + cx
    v_src = distorted_coords[..., 1] * fy + cy

    grid_sample_coords = torch.zeros_like(pixel_coords)
    grid_sample_coords[..., 0] = (u_src / (W - 1)) * 2 - 1
    grid_sample_coords[..., 1] = (v_src / (H - 1)) * 2 - 1

    undistorted = F.grid_sample(images, grid_sample_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
    return undistorted


def visualize_pose(image, points_2d, edges=None):
    vis_img = image.copy()
    for i, (x, y) in enumerate(points_2d):
        if np.isnan(x) or np.isnan(y): continue
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)
    if edges:
        for p1_idx, p2_idx in edges:
            if p1_idx >= len(points_2d) or p2_idx >= len(points_2d): continue
            pt1, pt2 = points_2d[p1_idx], points_2d[p2_idx]
            if np.isnan(pt1).any() or np.isnan(pt2).any(): continue
            cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
    return vis_img


def tensor_image_to_cv2(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8).copy()


@app.command()
def main(
        data: Path = typer.Option(..., help="Path to the data directory"),
        backbone_path: str = typer.Option("backbone"),
        stf_path: str = typer.Option("stf"),
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    L.seed_everything(42)

    backbone = framevision.autoloading.load_model(backbone_path, attribute="network").eval().cuda()
    stf = framevision.autoloading.load_model(stf_path, attribute="network").eval().cuda()
    model = Frame(backbone, stf).eval().cuda()

    timings = []
    undersampling_factor = getattr(stf, "undersampling_factor",
                                   stf._orig_mod.undersampling_factor if hasattr(stf, "_orig_mod") else 1)

    action_dir = data / "test_actor00_seq1" / "actions" / "archery"
    model.reset()
    frame_idx = 0
    video_writer = None
    output_path = "vis_output.mp4"

    for batch in tqdm(video_stream_generator(action_dir, None, device), desc="Streaming"):
        if frame_idx % undersampling_factor != 0:
            frame_idx += 1
            continue

        kwargs = unpack_batch_data(batch, device)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            joints3Dwr = model(**kwargs)

            with torch.no_grad():
                kwargs['images'] = undistort_images(kwargs['images'], kwargs['K'], kwargs['d'])

            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event)

        if model.is_warming_up():
            frame_idx += 1
            continue

        timings.append(duration)
        joints_np = joints3Dwr.squeeze().cpu().numpy()

        K_left = kwargs['K'][0].cpu().numpy()
        T_M_to_W = kwargs['middle2world'].cpu().numpy()
        T_L_to_M = kwargs['left2middle'].cpu().numpy()

        T_L_to_W = T_M_to_W @ T_L_to_M
        T_W_to_L = np.linalg.inv(T_L_to_W)

        kpts_2d = project_3d_to_2d(joints_np, K_left, T_W_to_L, width=256, height=256)

        vis_bg = tensor_image_to_cv2(kwargs['images'][0])
        vis_bg_bgr = cv2.cvtColor(vis_bg, cv2.COLOR_RGB2BGR)
        vis_result = visualize_pose(vis_bg_bgr, kpts_2d, edges=SKELETON_EDGES)

        if video_writer is None:
            h, w = vis_result.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        video_writer.write(vis_result)
        frame_idx += 1

    if video_writer: video_writer.release()

    if len(timings) > 2:
        durations = torch.tensor(sorted(timings))
        n = max(1, int(len(durations) * 0.1))
        durations = durations[n:-n]
        print(f"Mean duration: {durations.mean():.2f}ms ± {durations.std():.2f}ms")
    else:
        print("Not enough frames for statistics.")


def to_homogeneous(T):
    T = np.asarray(T)
    if T.shape[-2:] == (4, 4): return T
    H = np.eye(4, dtype=T.dtype)
    H[..., :T.shape[-2], :T.shape[-1]] = T
    return H


def video_stream_generator(action_dir: Path, processing, device):
    videos_dir = action_dir / "videos"
    left_imgs = sorted((videos_dir / "egocam_left").glob("*.jpg"))
    right_imgs = sorted((videos_dir / "egocam_right").glob("*.jpg"))
    meta_dir = action_dir.parents[1] / "meta"

    # --- 辅助函数：更健壮的畸变系数读取 ---
    def load_intrinsics_and_distortion(json_path):
        with open(json_path) as f:
            data = json.load(f)
        K = np.array(data["K"], dtype=np.float32)

        # 尝试多种 key 读取畸变系数
        keys_to_try = ["d", "distortion_coefficients", "dist_coeffs", "coeffs"]
        d = None
        for key in keys_to_try:
            if key in data:
                d = np.array(data[key], dtype=np.float32)
                break

        if d is None:
            # print(f"Warning: No distortion found in {json_path.name}, using zeros.")
            d = np.zeros(5, dtype=np.float32)

        return K, d

    K_l, d_l_raw = load_intrinsics_and_distortion(meta_dir / "intrinsics" / "egocam_left.json")
    K_r, d_r_raw = load_intrinsics_and_distortion(meta_dir / "intrinsics" / "egocam_right.json")

    l2m = np.load(meta_dir / "transforms" / "egocam_left_to_egocam_middle.npz")
    r2m = np.load(meta_dir / "transforms" / "egocam_right_to_egocam_middle.npz")

    left2middle_data = get_matrix_from_npz(l2m)
    right2middle_data = get_matrix_from_npz(r2m)

    left2middle = to_homogeneous(left2middle_data.astype(np.float32))
    right2middle = to_homogeneous(right2middle_data.astype(np.float32))

    camera_height = 1.6
    tilt_deg = 55
    theta = np.radians(tilt_deg)
    c, s = np.cos(theta), np.sin(theta)

    R_m2w = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)

    middle2world = np.eye(4, dtype=np.float32)
    middle2world[:3, :3] = R_m2w
    middle2world[1, 3] = camera_height

    cams2middle = np.stack([left2middle, right2middle], axis=0)

    TARGET_SIZE = (256, 256)

    for imgL_path, imgR_path in zip(left_imgs, right_imgs):
        imgL_raw = cv2.imread(str(imgL_path))[:, :, ::-1]
        imgR_raw = cv2.imread(str(imgR_path))[:, :, ::-1]

        h_orig, w_orig = imgL_raw.shape[:2]

        imgL_res = cv2.resize(imgL_raw, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        imgR_res = cv2.resize(imgR_raw, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        imgL_norm = (imgL_res.astype(np.float32) / 127.5) - 1.0
        imgR_norm = (imgR_res.astype(np.float32) / 127.5) - 1.0

        img_batch = np.stack([imgL_norm, imgR_norm], axis=0)
        img_tensor = torch.from_numpy(img_batch).permute(0, 3, 1, 2).to(device)

        K_scale = np.eye(3, dtype=np.float32)
        K_scale[0, 0] = TARGET_SIZE[0] / w_orig
        K_scale[1, 1] = TARGET_SIZE[1] / h_orig
        K_scale[0, 2] = TARGET_SIZE[0] / w_orig
        K_scale[1, 2] = TARGET_SIZE[1] / h_orig

        K_l_norm = K_scale @ K_l
        K_r_norm = K_scale @ K_r
        K_tensor = torch.from_numpy(np.stack([K_l_norm, K_r_norm], axis=0)).to(device)
        d_tensor = torch.from_numpy(np.stack([d_l_raw, d_r_raw], axis=0)).to(device)

        # ✅ 修复点：添加 transforms 键，防止 KeyError
        batch = {
            "images": img_tensor,
            "intrinsics_norm": {
                "K": K_tensor,
                "d": d_tensor,
            },
            "transforms": {
                "cams2middle": cams2middle,
                "middle2world": middle2world
            }
        }

        yield batch


def unpack_batch_data(batch, device):
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        return data.to(device)

    cams2middle = to_tensor(batch["transforms"]["cams2middle"])
    middle2world = to_tensor(batch["transforms"]["middle2world"])

    left2middle = cams2middle[0]
    right2middle = cams2middle[1]
    m2w = middle2world.reshape(4, 4)

    return dict(
        images=to_tensor(batch["images"]),
        K=to_tensor(batch["intrinsics_norm"]["K"]),
        d=to_tensor(batch["intrinsics_norm"]["d"]),
        left2middle=left2middle,
        right2middle=right2middle,
        middle2world=m2w,
    )


def get_matrix_from_npz(npz_archive):
    if 'rotations' in npz_archive and 'translations' in npz_archive:
        R = npz_archive['rotations']
        t = npz_archive['translations']
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.reshape(3, 3)
        T[:3, 3] = t.flatten()
        return T
    keys = npz_archive.files
    target_key = "T" if "T" in keys else keys[0]
    return npz_archive[target_key]


if __name__ == "__main__":
    app()