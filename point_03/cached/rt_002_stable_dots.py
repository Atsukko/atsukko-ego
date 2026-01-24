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

    out_dir = action_dir / "vis2d"
    (out_dir / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "right").mkdir(parents=True, exist_ok=True)

    model.reset()
    frame_idx = 0

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

        # --------------------------------------------------
        # 1. world -> camera
        # --------------------------------------------------
        joints_world = joints3Dwr.numpy()  # (J,3)

        # cam -> world
        left_cam2world = (
                batch["transforms"]["middle2world"].squeeze().cpu().numpy()
                @ batch["transforms"]["cams2middle"][0].cpu().numpy()
        )
        right_cam2world = (
                batch["transforms"]["middle2world"].squeeze().cpu().numpy()
                @ batch["transforms"]["cams2middle"][1].cpu().numpy()
        )

        # world -> cam
        world2left = np.linalg.inv(left_cam2world)
        world2right = np.linalg.inv(right_cam2world)

        def world_to_cam(joints, world2cam):
            J = joints.shape[0]
            joints_h = np.concatenate(
                [joints, np.ones((J, 1))], axis=1
            )  # (J,4)
            cam = (world2cam @ joints_h.T).T
            return cam[:, :3]

        joints_cam_L = world_to_cam(joints_world, world2left)
        joints_cam_R = world_to_cam(joints_world, world2right)
        #joints_cam_L[:, 0] *= -1
        #joints_cam_R[:, 0] *= -1
        # --------------------------------------------------
        # 2. camera -> image
        # --------------------------------------------------
        K_left = batch["intrinsics_norm"]["K"][0].cpu().numpy()
        K_right = batch["intrinsics_norm"]["K"][1].cpu().numpy()
        d_left = batch["intrinsics_norm"]["d"][0].cpu().numpy()
        d_right = batch["intrinsics_norm"]["d"][1].cpu().numpy()

        pts_L, _ = cv2.fisheye.projectPoints(
            joints_cam_L.reshape(-1, 1, 3).astype(np.float32),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K_left,
            d_left,
        )

        pts_R, _ = cv2.fisheye.projectPoints(
            joints_cam_R.reshape(-1, 1, 3).astype(np.float32),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K_right,
            d_right,
        )

        pts_L = pts_L.squeeze(1)
        pts_R = pts_R.squeeze(1)

        imgL = batch["images"][0].permute(1, 2, 0).cpu().numpy()
        imgR = batch["images"][1].permute(1, 2, 0).cpu().numpy()

        imgL = (imgL * 255).astype(np.uint8)
        imgR = (imgR * 255).astype(np.uint8)

        for x, y in pts_L:
            x, y = int(x), int(y)
            if 0 <= x < imgL.shape[1] and 0 <= y < imgL.shape[0]:
                cv2.circle(imgL, (x, y), 3, (0, 255, 0), -1)

        for x, y in pts_R:
            x, y = int(x), int(y)
            if 0 <= x < imgR.shape[1] and 0 <= y < imgR.shape[0]:
                cv2.circle(imgR, (x, y), 3, (0, 255, 0), -1)

        cv2.imwrite(
            str(out_dir / "left" / f"{frame_idx:06d}.jpg"),
            imgL[:, :, ::-1],
        )

        cv2.imwrite(
            str(out_dir / "right" / f"{frame_idx:06d}.jpg"),
            imgR[:, :, ::-1],
        )

        frame_idx += 1

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
