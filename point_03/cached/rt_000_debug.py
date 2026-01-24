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

# 骨架连接
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), (13, 14)
]


# --- ✅ 新增：旋转矩阵计算函数 ---
def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """ 计算相机的旋转矩阵 (ZYX顺序) """
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --- 视觉函数 (渐变色) ---
def get_gradient_color(index, total_points):
    ratio = index / (total_points - 1)
    return (int(255 * ratio), 0, int(255 * (1 - ratio)))


def visualize_pose(image, points_2d, edges=None):
    vis_img = image.copy()
    num_points = len(points_2d)
    if edges:
        for p1, p2 in edges:
            if p1 >= num_points or p2 >= num_points: continue
            pt1, pt2 = points_2d[p1], points_2d[p2]
            if np.isnan(pt1).any() or np.isnan(pt2).any(): continue
            color = get_gradient_color(p1, num_points)
            cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
    for i, (x, y) in enumerate(points_2d):
        if np.isnan(x) or np.isnan(y): continue
        color = get_gradient_color(i, num_points)
        cv2.circle(vis_img, (int(x), int(y)), 5, color, -1)
        cv2.circle(vis_img, (int(x), int(y)), 5, (255, 255, 255), 1)
    return vis_img


# --- 投影和去畸变基础函数 (保持不变) ---
def project_3d_to_2d(points_3d, K, T_world_to_cam, width=256, height=256):
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # 注意：如果输入的 points_3d 已经是相机坐标系，T_world_to_cam 必须是单位矩阵
    points_cam = (T_world_to_cam @ points_3d_h.T).T
    z = points_cam[:, 2]
    eps = 1e-6
    projected = (K @ points_cam[:, :3].T).T
    u, v = projected[:, 0] / (z + eps), projected[:, 1] / (z + eps)
    u[z <= 0], v[z <= 0] = np.nan, np.nan
    return np.stack([u, v], axis=1)


def distort(d: Float[Tensor, "B D"], normalized_coords: Float[Tensor, "B N 2"]) -> Float[Tensor, "B N 2"]:
    """
    修正后的畸变计算：确保 d 的维度可以与 normalized_coords (B, N, 2) 广播
    """
    # normalized_coords 形状是 (B, N, 2)
    r = torch.norm(normalized_coords, dim=-1, keepdim=True)  # (B, N, 1)
    theta = torch.atan(r)  # (B, N, 1)

    # 将 d 扩展为 (B, 1, D)，以便与 (B, N, 1) 的 theta 运算
    # 假设 d 的最后一维至少有 4 个系数 (k1, k2, k3, k4)
    d_fixed = d.unsqueeze(1)  # (B, 1, D)

    k1 = d_fixed[..., 0:1]  # (B, 1, 1)
    k2 = d_fixed[..., 1:2]
    k3 = d_fixed[..., 2:3]
    k4 = d_fixed[..., 3:4]

    # 计算畸变后的半径 r_d
    # theta 是 (B, N, 1)，k1 是 (B, 1, 1)，乘积是 (B, N, 1)
    r_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)

    # 防止除以 0
    scale = torch.where(r > 1e-8, r_d / r, torch.ones_like(r))
    distorted_coords = scale * normalized_coords  # (B, N, 2)

    return distorted_coords


def undistort_images(images, K, d):
    B, C, H, W = images.shape
    device = images.device

    # 生成网格
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # pixel_coords: (B, H*W, 2)
    pixel_coords = torch.stack([x, y], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)

    # 提取内参
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)

    # 归一化坐标 (B, N, 2)
    normalized = torch.zeros_like(pixel_coords)
    normalized[..., 0] = (pixel_coords[..., 0] - cx.squeeze(-1)) / fx.squeeze(-1)
    normalized[..., 1] = (pixel_coords[..., 1] - cy.squeeze(-1)) / fy.squeeze(-1)

    # 应用畸变模型
    distorted_flat = distort(d, normalized)

    # 映射回像素空间
    u_src = distorted_flat[..., 0] * fx.squeeze(-1) + cx.squeeze(-1)
    v_src = distorted_flat[..., 1] * fy.squeeze(-1) + cy.squeeze(-1)

    # 准备 grid_sample (要求范围在 [-1, 1])
    grid = torch.stack([
        (u_src / (W - 1)) * 2 - 1,
        (v_src / (H - 1)) * 2 - 1
    ], dim=-1).reshape(B, H, W, 2)

    return F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def tensor_image_to_cv2(t): return np.clip((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255).astype(
    np.uint8).copy()


@app.command()
def main(
        data: Path = typer.Option(...), backbone: str = "backbone", stf: str = "stf",
        warmup: int = typer.Option(50, help="手动设置更长的 warmup 帧数以稳定姿态"),
):
    device = torch.device("cuda")
    bb = framevision.autoloading.load_model(backbone, attribute="network").eval().cuda()
    stf_net = framevision.autoloading.load_model(stf, attribute="network").eval().cuda()
    model = Frame(bb, stf_net).eval().cuda()

    action_dir = data / "test_actor00_seq1" / "actions" / "archery"
    model.reset()
    f_idx, writer = 0, None
    gui_done = False  # ✅ 新增标志位
    undersample = getattr(stf_net, "undersampling_factor", 1)

    for batch in tqdm(video_stream_generator(action_dir, None, device), desc="Streaming"):
        if f_idx % undersample != 0: f_idx += 1; continue

        kwargs = unpack_batch_data(batch, device)
        with torch.no_grad():
            joints3Dwr = model(**kwargs)
            undistorted_imgs = undistort_images(kwargs['images'], kwargs['K'], kwargs['d'])

        # --- ✅ 修改 2：更长且更稳的 Warmup 逻辑 ---
        # 只有当模型自带 warmup 结束且达到我们手动设置的最小帧数时，才开始输出
        if model.is_warming_up() or f_idx < warmup:
            f_idx += 1
            continue

        # --- ✅ 修改 1：修正 RGB -> BGR 颜色问题 ---
        # 这里的 vis_bg 已经是 BGR 格式，方便后续 OpenCV 所有操作
        vis_bg_raw = tensor_image_to_cv2(undistorted_imgs[0])
        vis_bg = cv2.cvtColor(vis_bg_raw, cv2.COLOR_RGB2BGR)

        joints_world = joints3Dwr.squeeze().cpu().numpy()
        K_left = kwargs['K'][0].cpu().numpy()
        T_W2L = np.linalg.inv((kwargs['middle2world'] @ kwargs['left2middle']).cpu().numpy())

        # --- 🟢 弹出交互窗口 ---
        if not gui_done:
            window_name = "Tuning - Press Any Key to Finish"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # ... 创建滑块代码保持不变 ...
            cv2.createTrackbar("Yaw", window_name, 297, 360, lambda x: None)
            cv2.createTrackbar("Pitch", window_name, 231, 360, lambda x: None)
            cv2.createTrackbar("Roll", window_name, 114, 360, lambda x: None)
            cv2.createTrackbar("TX", window_name, 90, 200, lambda x: None)
            cv2.createTrackbar("TY", window_name, 90, 200, lambda x: None)
            cv2.createTrackbar("TZ", window_name, 89, 200, lambda x: None)

            while True:
                y_deg = cv2.getTrackbarPos("Yaw", window_name) - 180
                p_deg = cv2.getTrackbarPos("Pitch", window_name) - 180
                r_deg = cv2.getTrackbarPos("Roll", window_name) - 180
                tx = (cv2.getTrackbarPos("TX", window_name) - 100) * 0.05
                ty = (cv2.getTrackbarPos("TY", window_name) - 100) * 0.05
                tz = (cv2.getTrackbarPos("TZ", window_name) - 100) * 0.05

                y_rad, p_rad, r_rad = np.radians(y_deg), np.radians(p_deg), np.radians(r_deg)
                Ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
                Rx = np.array([[1, 0, 0], [0, np.cos(p_rad), -np.sin(p_rad)], [0, np.sin(p_rad), np.cos(p_rad)]])
                Rz = np.array([[np.cos(r_rad), -np.sin(r_rad), 0], [np.sin(r_rad), np.cos(r_rad), 0], [0, 0, 1]])
                R_total = Ry @ Rx @ Rz

                center = joints_world.mean(axis=0)
                pts = ((joints_world - center) @ R_total.T) + center
                pts += np.array([tx, ty, tz])

                kpts_2d = project_3d_to_2d(pts, K_left, T_W2L)
                # 直接传入 BGR 的 vis_bg
                vis_show = visualize_pose(vis_bg.copy(), kpts_2d, SKELETON_EDGES)

                cv2.imshow(window_name, vis_show)
                if cv2.waitKey(1) != -1:
                    final_params = {"yaw": y_deg, "pitch": p_deg, "roll": r_deg, "tx": tx, "ty": ty, "tz": tz}
                    gui_done = True
                    break
            cv2.destroyWindow(window_name)

        # --- 🔵 每一帧应用最终参数 ---
        y_f, p_f, r_f, tx_f, ty_f, tz_f = final_params.values()

        y_r, p_r, r_r = np.radians(y_f), np.radians(p_f), np.radians(r_f)
        Ry = np.array([[np.cos(y_r), 0, np.sin(y_r)], [0, 1, 0], [-np.sin(y_r), 0, np.cos(y_r)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p_r), -np.sin(p_r)], [0, np.sin(p_r), np.cos(p_r)]])
        Rz = np.array([[np.cos(r_r), -np.sin(r_r), 0], [np.sin(r_r), np.cos(r_r), 0], [0, 0, 1]])
        R_total_f = Ry @ Rx @ Rz

        center_curr = joints_world.mean(axis=0)
        pts_curr = ((joints_world - center_curr) @ R_total_f.T) + center_curr
        pts_curr += np.array([tx_f, ty_f, tz_f])

        kpts_2d_final = project_3d_to_2d(pts_curr, K_left, T_W2L)

        # --- 绘制并保存视频 ---
        # vis_bg 已经是 BGR 了，直接画即可
        vis_result = visualize_pose(vis_bg, kpts_2d_final, SKELETON_EDGES)

        if writer is None:
            h, w = vis_result.shape[:2]
            writer = cv2.VideoWriter("vis_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        writer.write(vis_result)
        f_idx += 1


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

    # 读取相机参数
    with open(meta_dir / "intrinsics" / "egocam_left.json") as f: K_l = np.array(json.load(f)["K"], dtype=np.float32)
    with open(meta_dir / "intrinsics" / "egocam_right.json") as f: K_r = np.array(json.load(f)["K"], dtype=np.float32)
    l2m = np.load(meta_dir / "transforms" / "egocam_left_to_egocam_middle.npz")
    r2m = np.load(meta_dir / "transforms" / "egocam_right_to_egocam_middle.npz")

    left2middle_data = get_matrix_from_npz(l2m)
    right2middle_data = get_matrix_from_npz(r2m)

    left2middle = to_homogeneous(left2middle_data.astype(np.float32))
    right2middle = to_homogeneous(right2middle_data.astype(np.float32))

    # --------------------------------------------------
    # ✅ 修正后的 middle2world: 模拟真实第一视角
    # --------------------------------------------------
    camera_height = 1.6  # 米
    tilt_deg = 55  # 向下倾斜角度
    theta = np.radians(tilt_deg)
    c, s = np.cos(theta), np.sin(theta)

    # 修正后的旋转矩阵：绕X轴旋转，使Z轴向下
    # R_middle2world 描述的是如何从相机坐标系转到世界坐标系
    R_m2w = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)

    middle2world = np.eye(4, dtype=np.float32)
    middle2world[:3, :3] = R_m2w
    middle2world[1, 3] = camera_height  # 设置相机高度

    cams2middle = np.stack([left2middle, right2middle], axis=0)

    # 假设你的模型输入是 256x256
    TARGET_SIZE = (256, 256)

    for imgL_path, imgR_path in zip(left_imgs, right_imgs):
        # 1. 读取原始图像 (BGR -> RGB)
        imgL_raw = cv2.imread(str(imgL_path))[:, :, ::-1]
        imgR_raw = cv2.imread(str(imgR_path))[:, :, ::-1]

        h_orig, w_orig = imgL_raw.shape[:2]

        # 2. 手动 Resize 图像
        imgL_res = cv2.resize(imgL_raw, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        imgR_res = cv2.resize(imgR_raw, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # 3. 手动归一化 (对应 NormalizeImages: (x/255.0 - 0.5) / 0.5 => x/127.5 - 1.0)
        imgL_norm = (imgL_res.astype(np.float32) / 127.5) - 1.0
        imgR_norm = (imgR_res.astype(np.float32) / 127.5) - 1.0

        img_batch = np.stack([imgL_norm, imgR_norm], axis=0)  # (2, 256, 256, 3)
        img_tensor = torch.from_numpy(img_batch).permute(0, 3, 1, 2).to(device)  # (2, 3, 256, 256)

        # 4. 手动缩放内参 K (对应 NormalizeIntrinsics)
        # 如果图像从 (h_orig, w_orig) 变成了 (256, 256)
        # 内参矩阵的 fx, fy, cx, cy 需要同步缩放
        K_scale = np.eye(3, dtype=np.float32)
        K_scale[0, 0] = TARGET_SIZE[0] / w_orig  # fx 缩放
        K_scale[1, 1] = TARGET_SIZE[1] / h_orig  # fy 缩放
        K_scale[0, 2] = TARGET_SIZE[0] / w_orig  # cx 缩放
        K_scale[1, 2] = TARGET_SIZE[1] / h_orig  # cy 缩放

        K_l_norm = K_scale @ K_l
        K_r_norm = K_scale @ K_r
        K_tensor = torch.from_numpy(np.stack([K_l_norm, K_r_norm], axis=0)).to(device)

        # 5. 构造最终 Batch
        batch = {
            "images": img_tensor,
            "intrinsics_norm": {
                "K": K_tensor,
                "d": torch.zeros((2, 5), device=device)
            },
            "transforms": {
                "cams2middle": torch.from_numpy(cams2middle).to(device),
                "middle2world": torch.from_numpy(middle2world[None]).to(device),
            },
        }

        yield batch


def unpack_batch_data(batch, device):
    # 辅助函数：确保数据是 Tensor
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        return data.to(device)

    # 1. 转换并处理位姿
    cams2middle = to_tensor(batch["transforms"]["cams2middle"])  # (2, 4, 4)
    middle2world = to_tensor(batch["transforms"]["middle2world"])  # (1, 4, 4)

    # 2. 提取左、右相机到位姿
    # 这里使用 [idx] 索引，对 numpy 或 tensor 都通用，不需要 unsqueeze
    left2middle = cams2middle[0]
    right2middle = cams2middle[1]

    # middle2world 形状通常是 (1, 4, 4)，需要去掉 Batch 维度变成 (4, 4)
    # 使用 .reshape(4, 4) 比 squeeze() 更安全
    m2w = middle2world.reshape(4, 4)

    return dict(
        images=to_tensor(batch["images"]),  # (2, 3, H, W)
        K=to_tensor(batch["intrinsics_norm"]["K"]),  # (2, 3, 3)
        d=to_tensor(batch["intrinsics_norm"]["d"]),  # (2, D)
        left2middle=left2middle,
        right2middle=right2middle,
        middle2world=m2w,
    )


def prepare_video_processing():
    return torchvision.transforms.Compose([
        framevision.processing.Resize((256, 256)),
        framevision.processing.NormalizeImages(),
        framevision.processing.NormalizeIntrinsics(),
    ])


def get_matrix_from_npz(npz_archive):
    # 方案 A: 检查是否存在分离的 rotations 和 translations
    if 'rotations' in npz_archive and 'translations' in npz_archive:
        R = npz_archive['rotations']  # 应该是 (3, 3)
        t = npz_archive['translations']  # 应该是 (3,) 或 (3, 1)

        T = np.eye(4, dtype=np.float32)
        # 确保 R 是 3x3，t 是 3 维向量
        T[:3, :3] = R.reshape(3, 3)
        T[:3, 3] = t.flatten()
        return T

    # 方案 B: 检查是否存在直接的变换矩阵 T
    keys = npz_archive.files
    target_key = "T" if "T" in keys else keys[0]
    return npz_archive[target_key]


if __name__ == "__main__":
    app()