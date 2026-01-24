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

import cv2
import numpy as np
import torch

# 这是一个通用的骨架连接定义（假设是 Human3.6M 或类似拓扑，你需要根据你的模型输出调整）
# 如果不确定，可以先设为空列表 []，只看点
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),  # 左臂链
    (0, 4), (4, 5), (5, 6),  # 右臂链
    (0, 7), (7, 8), (8, 9), (9, 10),  # 左腿链
    (0, 11), (11, 12), (12, 13), (13, 14)  # 右腿链
]


def project_3d_to_2d(points_3d, K, T_world_to_cam):
    """
    将世界坐标系下的 3D 点投影到像素坐标系
    points_3d: (J, 3) numpy array
    K: (3, 3) 内参矩阵
    T_world_to_cam: (4, 4) 世界到相机的变换矩阵 (即 cam2world 的逆)
    """
    # 1. 转为齐次坐标 (J, 4)
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_h = np.hstack([points_3d, ones])

    # 2. 变换到相机坐标系: P_cam = T * P_world
    # 注意矩阵乘法顺序，取决于你的点是行向量还是列向量
    # 这里假设 T 是标准的 (4,4), points_3d_h 转置后相乘
    points_cam = (T_world_to_cam @ points_3d_h.T).T  # (J, 4)

    # 取出 XYZ
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # 3. 透视投影: u = fx * X/Z + cx
    # 使用矩阵乘法: p_2d_h = K @ [X, Y, Z]
    points_cam_xyz = points_cam[:, :3]
    projected = (K @ points_cam_xyz.T).T  # (J, 3)

    # 归一化 (除以 Z)
    u = projected[:, 0] / (projected[:, 2] + 1e-8)
    v = projected[:, 1] / (projected[:, 2] + 1e-8)

    return np.stack([u, v], axis=1)


def visualize_pose(image, points_2d, edges=None):
    """
    在图像上画点和骨架
    image: HxWxC numpy array (uint8)
    points_2d: (J, 2)
    """
    vis_img = image.copy()

    # 画点
    for i, (x, y) in enumerate(points_2d):
        if np.isnan(x) or np.isnan(y): continue
        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)  # 红色点

    # 画骨架连线
    if edges:
        for p1_idx, p2_idx in edges:
            if p1_idx >= len(points_2d) or p2_idx >= len(points_2d): continue

            pt1 = points_2d[p1_idx]
            pt2 = points_2d[p2_idx]

            if np.isnan(pt1).any() or np.isnan(pt2).any(): continue

            cv2.line(vis_img,
                     (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])),
                     (0, 255, 0), 2)  # 绿色线
    return vis_img


def tensor_image_to_cv2(tensor_img):
    """
    将网络输入的 Tensor (C, H, W) 转回 OpenCV 格式 (H, W, C) 并反归一化
    假设输入已经 Normalize 过了，这里简化处理，直接以此为底图可视化
    如果你的 Normalize 包含 mean/std 偏移，这里最好做对应的 denormalize
    """
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    # 假设输入范围大致在 -2~2 或 0~1 之间，将其拉伸到 0-255
    # 最简单的可视化方法：MinMax 归一化
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.uint8).copy()

@app.command()
def main(
    data: Path = typer.Option(
        ..., help="Path to the data directory"
    ),
    backbone_path: str = typer.Option(
        "backbone",
        help="Backbone to use for evaluation. Name of the exp or the path to the checkpoint.",
    ),
    stf_path: str = typer.Option(
        "stf",
        help="STF to use for evaluation. Can be the name of the experiment or the path to the checkpoint.",
    ),
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run this script on a machine with an NVIDIA GPU."
        )

    device = torch.device("cuda")
    L.seed_everything(42)

    backbone = (
        framevision.autoloading
        .load_model(backbone_path, attribute="network")
        .eval()
        .cuda()
    )
    stf = (
        framevision.autoloading
        .load_model(stf_path, attribute="network")
        .eval()
        .cuda()
    )

    backbone = torch.compile(
        backbone, mode="max-autotune", fullgraph=True
    )
    stf = torch.compile(
        stf, mode="max-autotune", fullgraph=True
    )

    model = Frame(backbone, stf).eval().cuda()

    processing = prepare_video_processing()
    timings = []

    undersampling_factor = (
        stf.undersampling_factor
        if hasattr(stf, "undersampling_factor")
        else stf._orig_mod.undersampling_factor
    )

    # 选一个 action 当视频
    action_dir = (
        data
        / "test_actor00_seq1"
        / "actions"
        / "archery"  # 你可以换
    )

    model.reset()
    frame_idx = 0

    # 1. 定义视频写入器 (初始设为 None，等拿到第一帧图片确定大小时再初始化)
    video_writer = None
    output_video_path = "vis_output.mp4"

    for batch in tqdm(
            video_stream_generator(action_dir, processing, device),
            desc="Streaming video frames",
    ):
        # undersampling
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

    for batch in tqdm(
        video_stream_generator(action_dir, processing, device),
        desc="Streaming video frames",
    ):
        # undersampling（必须保留）
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

        joints3Dwr = joints3Dwr.squeeze().cpu()
        timings.append(duration)

        # ✅ 到这里就成功了
        # print(joints3Dwr.shape)  # (J, 3)

        # ==========================================
        # ✅ 可视化并保存视频的核心代码
        # ==========================================

        # 1. 准备数据
        kpts_3d = joints3Dwr.numpy()

        # 反算变换矩阵: World -> Left Camera
        T_left2world = kwargs['left2middle'].cpu().numpy()
        T_world2left = np.linalg.inv(T_left2world)

        # 左目内参
        K_left = kwargs['K'][0].cpu().numpy()

        # 2. 投影 3D -> 2D (调用之前给你的 project_3d_to_2d 函数)
        kpts_2d = project_3d_to_2d(kpts_3d, K_left, T_world2left)

        # 3. 准备背景图 (Tensor -> Numpy)
        img_left_tensor = kwargs['images'][0]
        vis_bg = tensor_image_to_cv2(img_left_tensor)  # 调用之前给你的 tensor_image_to_cv2

        # 4. 绘制 (调用之前给你的 visualize_pose)
        # 这里的 edges 暂时设为 None，如果你有骨架连接列表，请替换 None
        vis_result = visualize_pose(vis_bg, kpts_2d, edges=None)

        # 5. 写入视频帧 (替代 imshow)
        if video_writer is None:
            # 初始化视频写入器
            h, w = vis_result.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID'
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))

        video_writer.write(vis_result)
        # ==========================================

        frame_idx += 1

    # 循环结束后关闭 writer
    if video_writer is not None:
        video_writer.release()
        print(f"Visualization saved to {output_video_path}")

    # Remove top 10% and bottom 10% of durations
    durations = torch.tensor(sorted(timings))
    n = int(len(durations) * 0.1)
    durations = durations[n:-n]

    mean_duration = durations.mean()
    std_duration = durations.std()

    print(
        f"Mean duration: {mean_duration:.2f}ms ± {std_duration:.2f}ms"
    )


def to_homogeneous(T):
    """
    Convert rotation / RT matrix to homogeneous form.
    Supports shapes:
      (..., 3, 3)
      (..., 3, 4)
      (..., 4, 4)
    Returns:
      (..., 4, 4)
    """
    T = np.asarray(T)

    if T.shape[-2:] == (4, 4):
        return T

    elif T.shape[-2:] == (3, 3):
        H = np.zeros(T.shape[:-2] + (4, 4), dtype=T.dtype)
        H[..., :3, :3] = T
        H[..., 3, 3] = 1.0
        return H

    elif T.shape[-2:] == (3, 4):
        H = np.zeros(T.shape[:-2] + (4, 4), dtype=T.dtype)
        H[..., :3, :4] = T
        H[..., 3, 3] = 1.0
        return H

    else:
        raise ValueError(f"Unsupported transform shape: {T.shape}")



def video_stream_generator(action_dir: Path, processing, device):
    """
    从 dataset/actions/action_xx/videos 中读取双目图像，
    并从 meta 中读取 intrinsics / transforms，
    构造可直接喂给 Frame 的 batch
    """
    # --------------------------------------------------
    # 1. paths
    # --------------------------------------------------
    videos_dir = action_dir / "videos"
    left_dir = videos_dir / "egocam_left"
    right_dir = videos_dir / "egocam_right"

    left_imgs = sorted(left_dir.glob("*.jpg"))
    right_imgs = sorted(right_dir.glob("*.jpg"))

    assert len(left_imgs) == len(right_imgs), \
        "Left/Right image count mismatch"

    meta_dir = action_dir.parents[1] / "meta"

    # --------------------------------------------------
    # 2. load intrinsics (json)
    # --------------------------------------------------
    with open(meta_dir / "intrinsics" / "egocam_left.json", "r") as f:
        intr_left = json.load(f)

    with open(meta_dir / "intrinsics" / "egocam_right.json", "r") as f:
        intr_right = json.load(f)

    # ⚠️ 假设字段名（如不一致，改这里）
    K_left = np.array(intr_left["K"], dtype=np.float32)
    K_right = np.array(intr_right["K"], dtype=np.float32)

    d_left = np.array(
        intr_left.get("d", intr_left.get("dist", [0, 0, 0, 0, 0])),
        dtype=np.float32,
    )
    d_right = np.array(
        intr_right.get("d", intr_right.get("dist", [0, 0, 0, 0, 0])),
        dtype=np.float32,
    )

    # --------------------------------------------------
    # 3. load transforms (npz)
    # --------------------------------------------------
    l2m = np.load(
        meta_dir / "transforms" / "egocam_left_to_egocam_middle.npz"
    )
    r2m = np.load(
        meta_dir / "transforms" / "egocam_right_to_egocam_middle.npz"
    )

    left2middle = l2m["T"] if "T" in l2m else l2m[list(l2m.keys())[0]]
    right2middle = r2m["T"] if "T" in r2m else r2m[list(r2m.keys())[0]]

    left2middle = to_homogeneous(left2middle.astype(np.float32))
    right2middle = to_homogeneous(right2middle.astype(np.float32))

    # --------------------------------------------------
    # 4. middle2world
    # --------------------------------------------------
    middle2world = np.eye(4, dtype=np.float32)

    cams2middle = np.stack(
        [left2middle, right2middle],
        axis=0,
    ).astype(np.float32)  # (2, 4, 4)

    # --------------------------------------------------
    # 5. per-frame loop
    # --------------------------------------------------
    for imgL_path, imgR_path in zip(left_imgs, right_imgs):
        imgL = cv2.imread(str(imgL_path))[:, :, ::-1]
        imgR = cv2.imread(str(imgR_path))[:, :, ::-1]

        batch = {
            "images": np.stack([imgL, imgR], axis=0),
            "intrinsics_norm": {
                "K": np.stack([K_left, K_right], axis=0),
                "d": np.stack([d_left, d_right], axis=0),
            },
            "transforms": {
                "cams2middle": cams2middle,
                "middle2world": middle2world[None],
            },
        }

        # --------------------------------------------------
        # 6. numpy -> torch
        # --------------------------------------------------
        batch["images"] = (
            torch.from_numpy(batch["images"])
            .permute(0, 3, 1, 2)
            .float()
            .to(device)
        )
        batch["intrinsics_norm"]["K"] = (
            torch.from_numpy(batch["intrinsics_norm"]["K"])
            .float()
            .to(device)
        )
        batch["intrinsics_norm"]["d"] = (
            torch.from_numpy(batch["intrinsics_norm"]["d"])
            .float()
            .to(device)
        )
        batch["transforms"]["cams2middle"] = (
            torch.from_numpy(batch["transforms"]["cams2middle"])
            .float()
            .to(device)
        )
        batch["transforms"]["middle2world"] = (
            torch.from_numpy(batch["transforms"]["middle2world"])
            .float()
            .to(device)
        )

        yield batch


def unpack_batch_data(batch, device):
    cams2middle = batch["transforms"]["cams2middle"]  # (2, ?, 4, 4) or (2,4,4)

    left2middle = cams2middle[0].squeeze()   # (4,4) ✅
    right2middle = cams2middle[1].squeeze()  # (4,4) ✅

    middle2world = batch["transforms"]["middle2world"].squeeze()  # (4,4)

    return dict(
        images=batch["images"].squeeze().to(device),          # (2,3,H,W)
        K=batch["intrinsics_norm"]["K"].squeeze().to(device), # (2,3,3)
        d=batch["intrinsics_norm"]["d"].squeeze().to(device), # (2,D)

        left2middle=left2middle.to(device),
        right2middle=right2middle.to(device),
        middle2world=middle2world.to(device),
    )



def prepare_video_processing():
    return torchvision.transforms.Compose(
        [
            framevision.processing.Resize((256, 256)),
            framevision.processing.NormalizeImages(),
            framevision.processing.NormalizeIntrinsics(),
        ]
    )


if __name__ == "__main__":
    app()
