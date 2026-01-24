from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torchvision
import typer
import lightning as L
from tqdm import tqdm

import framevision
from framevision.model import Frame
from framevision.pl_wrappers import FrameDataModule

app = typer.Typer()

# ✅ 修正后的骨架连接（适用于 FRAME 模型常用的 15/17 点拓扑）
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),  # 左臂链
    (0, 4), (4, 5), (5, 6),  # 右臂链
    (0, 7), (7, 8), (8, 9), (9, 10),  # 左腿链
    (0, 11), (11, 12), (12, 13), (13, 14)  # 右腿链
]


def project_3d_to_2d(points_3d, K, T_world_to_cam, width=256, height=256):
    # 1. 转换到相机空间
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_cam = (T_world_to_cam @ points_3d_h.T).T

    z = points_cam[:, 2]
    eps = 1e-6

    # 2. 投影计算
    # 注意：如果 K 已经是针对 256x256 缩放过的，这里直接矩阵相乘即可得到像素坐标
    projected = (K @ points_cam[:, :3].T).T

    u = projected[:, 0] / (z + eps)
    v = projected[:, 1] / (z + eps)

    # 3. 过滤
    mask = z <= 0
    u[mask] = np.nan
    v[mask] = np.nan

    # ⚠️ 重要：如果 K 已经包含像素单位（缩放后），就不需要再乘以 width/height
    return np.stack([u, v], axis=1)


def visualize_pose(image, points_2d, edges=None):
    vis_img = image.copy()
    # 画点
    for i, (x, y) in enumerate(points_2d):
        if np.isnan(x) or np.isnan(y): continue
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)
    # 画线
    if edges:
        for p1_idx, p2_idx in edges:
            if p1_idx >= len(points_2d) or p2_idx >= len(points_2d): continue
            pt1, pt2 = points_2d[p1_idx], points_2d[p2_idx]
            if np.isnan(pt1).any() or np.isnan(pt2).any(): continue
            cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
    return vis_img


def tensor_image_to_cv2(tensor_img):
    """ 将 (-1, 1) 的 Tensor 转回 (0, 255) 的 RGB 图像 """
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) * 127.5  # 反归一化
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

    # 加载模型
    backbone = framevision.autoloading.load_model(backbone_path, attribute="network").eval().cuda()
    stf = framevision.autoloading.load_model(stf_path, attribute="network").eval().cuda()
    model = Frame(backbone, stf).eval().cuda()

    timings = []
    undersampling_factor = getattr(stf, "undersampling_factor",
                                   stf._orig_mod.undersampling_factor if hasattr(stf, "_orig_mod") else 1)

    action_dir = data / "test_actor00_seq1" / "actions" / "avoid_some_bullets"
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

            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event)

        if model.is_warming_up():
            frame_idx += 1
            continue

        timings.append(duration)
        joints_np = joints3Dwr.squeeze().cpu().numpy()

        # --- 投影逻辑 ---
        # 1. 获取相机相对于 World 的位姿
        K_left = kwargs['K'][0].cpu().numpy()
        T_M_to_W = kwargs['middle2world'].cpu().numpy()
        T_L_to_M = kwargs['left2middle'].cpu().numpy()

        # 2. 计算左相机的 World->Cam 变换
        T_L_to_W = T_M_to_W @ T_L_to_M
        T_W_to_L = np.linalg.inv(T_L_to_W)

        # 3. 投影到 2D
        kpts_2d = project_3d_to_2d(joints_np, K_left, T_W_to_L, width=256, height=256)

        # --- 可视化 ---
        vis_bg = tensor_image_to_cv2(kwargs['images'][0])
        # 将 RGB 转回 BGR 供 cv2 绘制和写入
        vis_bg_bgr = cv2.cvtColor(vis_bg, cv2.COLOR_RGB2BGR)

        vis_result = visualize_pose(vis_bg_bgr, kpts_2d, edges=SKELETON_EDGES)

        if video_writer is None:
            h, w = vis_result.shape[:2]

            # --- 方案 A: 使用 mp4v (通常 Windows 自带支持) ---
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # --- 方案 B: 如果 A 还报错，尝试这个 (文件名后缀改为 .avi) ---
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # output_path = "vis_output.avi"

            video_writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        video_writer.write(vis_result)

        # # --- 调试打印开始 ---
        # if frame_idx % 30 == 0:  # 每30帧打一次，避免刷屏
        #     print(f"\n--- Frame {frame_idx} Debug Info ---")
        #     # 1. 检查模型输出的 3D 点范围
        #     print(f"Joints3D (Model Output) - Max: {joints_np.max(axis=0)}, Min: {joints_np.min(axis=0)}")
        #
        #     # 2. 检查相机相对于世界的坐标 (判断相机是不是在地面以下或者飞得太高)
        #     cam_pos_in_world = T_L_to_W[:3, 3]
        #     print(f"Camera Pos in World: {cam_pos_in_world}")
        #
        #     # 3. 检查 3D 点在相机坐标系下的位置 (Z 应该是正数，且在 0.5~3.0 之间)
        #     points_3d_h = np.hstack([joints_np, np.ones((joints_np.shape[0], 1))])
        #     points_cam = (T_W_to_L @ points_3d_h.T).T
        #     print(f"Joints in Camera Space (Z depth) - Mean: {points_cam[:, 2].mean():.2f}")
        #
        #     # 4. 检查投影后的 2D 坐标 (正常应在 0-255 之间)
        #     print(f"Projected 2D (First 3 joints):\n{kpts_2d[:3]}")
        # # --- 调试打印结束 ---

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