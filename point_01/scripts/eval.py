from pathlib import Path
from typing import Dict

import lightning as L
import torch
import typer
from tqdm import tqdm
from utils import resume_experiment_config

from framevision import geometry as geo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer()

#torch.set_float32_matmul_precision('high')
@app.command()
def main(
    data: Path = typer.Option(..., help="Path to the data directory"),
    load: str = typer.Option(..., help="Checkpoint to load. Can be the name of the experiment or the path to the checkpoint."),
):
    L.seed_everything(42)

    hparams, objects = resume_experiment_config(load, data, instantiate_objects=True)

    model = objects.model.network.to(device).eval()
    datamodule = objects.dataset
    datamodule.batch_size = 32
    datamodule.setup()

    preds, gt = [], []
    for batch in tqdm(datamodule.val_dataloader(), desc="Processing batches"):
        kwargs = unpack_batch_data(batch, device) if "images" in batch else unpack_cached_batch_data(batch, device)
        joints3Dgt = batch["body_tracking"]["joints_3D"].to(device, non_blocking=True)

        with torch.no_grad():
            output = model(**kwargs)

        joints3Dwr = get_world_prediction(output, batch).squeeze(1).cpu().half()
        joints3Dgt = joints3Dgt[:, -1:].squeeze(1).cpu().half()

        preds.append(joints3Dwr)
        gt.append(joints3Dgt)

    pred_tensor, gt_tensor = torch.cat(preds), torch.cat(gt)
    errors = pred_tensor - gt_tensor

    mpjpe = torch.norm(errors, dim=-1).mean() * 1000
    print(f"MPJPE: {mpjpe:.2f}mm")

    pck = (torch.norm(errors, dim=-1) < 0.1).float().mean() * 100
    print(f"3D-PCK: {pck:.2f}%")

    pa_mpjpe = compute_pampjpe(gt_tensor, pred_tensor).mean() * 1000
    print(f"PA-MPJPE: {pa_mpjpe:.2f}mm")


def compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (B x J x 3) closest to a set of 3D points S2,
    where R is a 3x3 rotation matrix, t is a 3x1 translation, and s is the scale.
    """
    batch_size, num_joints, _ = S1.shape

    # 1. Remove mean.
    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=(1, 2), keepdim=True)

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1.transpose(1, 2), X2)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, V.transpose(1, 2))))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.transpose(1, 2))

    # 5. Recover scale.
    scale = torch.sum(R * K, dim=(1, 2), keepdim=True) / var1

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(mu1, R.transpose(1, 2))

    # 7. Transform S1.
    S1_hat = scale * torch.matmul(S1, R.transpose(1, 2)) + t

    return S1_hat


def align_by_pelvis_torch(joints):
    """
    Assumes joints is B x J x 3.
    Aligns joints by subtracting the midpoint of the left and right hips.
    """
    left_id = 7
    right_id = 11

    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.0
    return joints - pelvis[:, None, :]


def compute_pampjpe(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: B x J x 3
      - preds: B x J x 3
    """
    # Align by pelvis.
    gt3d_aligned = align_by_pelvis_torch(gt3ds).float()
    pred3d_aligned = align_by_pelvis_torch(preds).float()

    # Get PA-MPJPE.
    pred3d_sym = compute_similarity_transform_torch(pred3d_aligned, gt3d_aligned)
    pa_error = torch.sqrt(torch.sum((gt3d_aligned - pred3d_sym) ** 2, dim=2))
    pa_mpjpe = torch.mean(pa_error, dim=1)

    return pa_mpjpe


def unpack_batch_data(batch, device):
    images = batch["images"].to(device, non_blocking=True)
    K = batch["intrinsics_norm"]["K"].to(device, non_blocking=True)
    d = batch["intrinsics_norm"]["d"].to(device, non_blocking=True)
    return dict(images=images, K=K, d=d)


def unpack_cached_batch_data(batch, device):
    joints_3D_cc = batch["cache"]["joints_3D_cc"].to(device, non_blocking=True)
    lTm = batch["transforms"]["egocam_left_to_egocam_middle"].to(device, non_blocking=True)
    rTm = batch["transforms"]["egocam_right_to_egocam_middle"].to(device, non_blocking=True)
    mTw = batch["poses"]["vr"]["egocam_middle"].to(device, non_blocking=True)
    return dict(joints_3D_cc=joints_3D_cc, left2middle=lTm, right2middle=rTm, middle2world=mTw)


def get_world_prediction(prediction: Dict, batch: Dict):
    if "joints_3D" in prediction:
        joints3Dwr = prediction["joints_3D"]  # Shape: (B, T, J, 3)
    elif "joints_3D_cc" in prediction:
        # We only evaluate the predictions from the left camera
        joints3Dcc = prediction["joints_3D_cc"]
        cam_poses = batch["cam_poses"]["vr"].to(joints3Dcc.device)
        joints3Dwr = geo.rototranslate(joints3Dcc, cam_poses)[:, :, 0]  # Shape: (B, T, J, 3)
    else:
        raise ValueError(f"No valid key found in prediction: {prediction.keys()}")

    return joints3Dwr


if __name__ == "__main__":
    app()
