from pathlib import Path
from typing import Dict
import time
import numpy as np

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from tqdm import tqdm
from utils import resume_experiment_config

from framevision import geometry as geo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer()


# torch.set_float32_matmul_precision('high')
@app.command()
def main(
        data: Path = typer.Option(..., help="Path to the data directory"),
        load: str = typer.Option(...,
                                 help="Checkpoint to load. Can be the name of the experiment or the path to the checkpoint."),
        warmup_iters: int = typer.Option(10, help="Number of warmup iterations for timing"),
        timing_iters: int = typer.Option(100, help="Number of iterations for timing measurement"),
):
    L.seed_everything(42)

    hparams, objects = resume_experiment_config(load, data, instantiate_objects=True)

    model = objects.model.network.to(device).eval()
    datamodule = objects.dataset
    datamodule.batch_size = 32
    datamodule.setup()

    # 推理时间测量
    print("=== 推理时间测量 ===")
    inference_time = measure_inference_time(model, datamodule, device, warmup_iters, timing_iters)
    print(f"平均推理时间: {inference_time['mean']:.2f}ms ± {inference_time['std']:.2f}ms")
    print(f"FPS: {inference_time['fps']:.2f}")
    print(f"峰值内存使用: {inference_time['peak_memory']:.2f}MB")
    print(f"模型参数量: {inference_time['params'] / 1e6:.2f}M")
    print(f"FLOPs: {inference_time['flops'] / 1e9:.2f}G\n")

    # 原有的精度评估
    preds, gt = [], []
    all_joints_3D_wr = []  # 存储所有帧的预测结果用于计算jitter

    for batch in tqdm(datamodule.val_dataloader(), desc="Processing batches"):
        kwargs = unpack_batch_data(batch, device) if "images" in batch else unpack_cached_batch_data(batch, device)
        joints3Dgt = batch["body_tracking"]["joints_3D"].to(device, non_blocking=True)

        with torch.no_grad():
            output = model(**kwargs)

        joints3Dwr = get_world_prediction(output, batch).squeeze(1).cpu().half()
        joints3Dgt = joints3Dgt[:, -1:].squeeze(1).cpu().half()

        preds.append(joints3Dwr)
        gt.append(joints3Dgt)

        # 存储所有帧的预测用于jitter计算
        if "all_joints_3D" in output:
            all_joints_3D_wr.append(output["all_joints_3D"].cpu().half())

    pred_tensor, gt_tensor = torch.cat(preds), torch.cat(gt)
    errors = pred_tensor - gt_tensor

    # 基础指标
    mpjpe = torch.norm(errors, dim=-1).mean() * 1000
    print(f"MPJPE: {mpjpe:.2f}mm")

    pck = (torch.norm(errors, dim=-1) < 0.1).float().mean() * 100
    print(f"3D-PCK: {pck:.2f}%")

    pa_mpjpe = compute_pampjpe(gt_tensor, pred_tensor).mean() * 1000
    print(f"PA-MPJPE: {pa_mpjpe:.2f}mm")

    # 新增指标计算
    print("\n=== 扩展评估指标 ===")

    # 1. Jitter (关节抖动) - 需要时序数据
    if all_joints_3D_wr:
        jitter_score = compute_jitter(all_joints_3D_wr)
        print(f"Jitter: {jitter_score:.4f}")

    # 2. NPP (Normalized Prediction Power)
    npp_score = compute_npp(pred_tensor, gt_tensor)
    print(f"NPP: {npp_score:.4f}")

    # 3. MPE (Mean Position Error) - 各关节平均误差
    mpe_scores = compute_mpe(pred_tensor, gt_tensor)
    print(f"MPE (各关节平均误差mm): {mpe_scores.mean():.2f}mm")
    print(f"MPE 各关节详情: {['%.1f' % x for x in mpe_scores.tolist()]}")

    # 4. FS (F-Score) - 基于距离阈值的F1分数
    fs_score = compute_fscore(pred_tensor, gt_tensor, threshold=0.1)
    print(f"FS@0.1m: {fs_score:.4f}")

    # 5. 额外的PCK阈值
    for threshold in [0.05, 0.15, 0.2]:
        pck_at_thresh = (torch.norm(errors, dim=-1) < threshold).float().mean() * 100
        print(f"PCK@{threshold * 100:.0f}cm: {pck_at_thresh:.2f}%")

    # 6. 平均每关节误差统计
    joint_errors = torch.norm(errors, dim=-1)  # [B*T, J]
    mean_per_joint = joint_errors.mean(dim=0) * 1000
    std_per_joint = joint_errors.std(dim=0) * 1000
    print(f"\n各关节误差统计 (mm):")
    print(f"平均值: {mean_per_joint.mean():.2f} ± {mean_per_joint.std():.2f}")
    print(f"中位数: {torch.median(mean_per_joint):.2f}")
    print(f"最大值: {mean_per_joint.max():.2f} (关节{mean_per_joint.argmax().item()})")
    print(f"最小值: {mean_per_joint.min():.2f} (关节{mean_per_joint.argmin().item()})")


def measure_inference_time(model, datamodule, device, warmup_iters=10, timing_iters=100):
    """测量模型推理时间、内存使用和计算量"""
    results = {}

    # 获取一个样本批次用于测量
    dataloader = datamodule.val_dataloader()
    sample_batch = next(iter(dataloader))

    # 准备输入数据
    if "images" in sample_batch:
        kwargs = unpack_batch_data(sample_batch, device)
    else:
        kwargs = unpack_cached_batch_data(sample_batch, device)

    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    print("Warming up...")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 正式计时
    print("Measuring inference time...")
    times = []
    with torch.no_grad():
        for i in range(timing_iters):
            start_time = time.time()

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            _ = model(**kwargs)

            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # 毫秒
            else:
                elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒

            times.append(elapsed_time)

    # 计算统计量
    times = np.array(times)
    results['mean'] = np.mean(times)
    results['std'] = np.std(times)
    results['min'] = np.min(times)
    results['max'] = np.max(times)
    results['fps'] = 1000 / results['mean']  # FPS

    # 内存使用
    if torch.cuda.is_available():
        results['peak_memory'] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        results['peak_memory'] = 0

    # 模型参数量
    results['params'] = sum(p.numel() for p in model.parameters())

    # 估算FLOPs (简化版本)
    results['flops'] = estimate_flops(model, kwargs)

    return results


def estimate_flops(model, sample_input):
    """估算模型的FLOPs (简化版本)"""
    # 这是一个简化的FLOPs估算，实际应该使用torchprofile等工具
    # 这里我们基于模型结构做一个粗略估算

    total_flops = 0
    batch_size = next(iter(sample_input.values())).shape[0] if sample_input else 1

    # 遍历模型的所有模块进行估算
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # FLOPs for Linear: 2 * in_features * out_features * batch_size
            if hasattr(module, 'weight'):
                in_features, out_features = module.weight.shape
                total_flops += 2 * in_features * out_features * batch_size

        elif isinstance(module, nn.Conv2d):
            # FLOPs for Conv2d: 2 * kernel_h * kernel_w * in_ch * out_ch * out_h * out_w * batch_size
            if hasattr(module, 'weight'):
                out_ch, in_ch, k_h, k_w = module.weight.shape
                # 估算输出特征图尺寸
                # 这是一个简化，实际应该计算真实的输出尺寸
                out_h, out_w = 32, 32  # 假设的输出尺寸
                total_flops += 2 * k_h * k_w * in_ch * out_ch * out_h * out_w * batch_size

        elif isinstance(module, nn.MultiheadAttention):
            # Transformer attention FLOPs估算
            # 这是一个简化版本
            seq_len = sample_input.get('joints_3D_cc', torch.randn(batch_size, 10, 17, 3)).shape[1]
            embed_dim = module.embed_dim
            # 简化的注意力FLOPs估算
            total_flops += 4 * batch_size * seq_len * seq_len * embed_dim

    return total_flops


def compute_jitter(all_predictions):
    """计算关节抖动 - 连续帧间位置变化的标准差"""
    if not all_predictions:
        return 0.0

    # 合并所有batch的预测 [B, T, J, 3]
    all_preds = torch.cat([pred for pred in all_predictions], dim=0)
    B, T, J, C = all_preds.shape

    if T < 2:
        return 0.0

    # 计算连续帧间的位移
    displacements = []
    for t in range(1, T):
        disp = torch.norm(all_preds[:, t] - all_preds[:, t - 1], dim=-1)  # [B, J]
        displacements.append(disp)

    if not displacements:
        return 0.0

    displacements = torch.stack(displacements, dim=0)  # [T-1, B, J]

    # 计算位移的标准差作为抖动指标
    jitter = displacements.std()
    return jitter.item()


def compute_npp(pred, gt, alpha=0.5):
    """
    计算归一化预测能力 (Normalized Prediction Power)
    结合位置误差和方向误差
    """
    B, J, C = pred.shape

    # 位置误差 (MPJPE)
    pos_error = torch.norm(pred - gt, dim=-1).mean()

    # 方向误差 - 使用骨骼向量计算
    # 假设关节顺序是标准的，计算相邻关节形成的骨骼方向
    bone_errors = []

    # 常见的骨骼连接 (需要根据你的关节顺序调整)
    # 这里使用一个通用的连接假设，你可能需要根据你的数据集调整
    bone_pairs = []
    for i in range(J - 1):
        bone_pairs.append((i, i + 1))

    for j1, j2 in bone_pairs:
        bone_pred = pred[:, j2] - pred[:, j1]  # [B, 3]
        bone_gt = gt[:, j2] - gt[:, j1]  # [B, 3]

        # 计算方向余弦相似度
        cos_sim = F.cosine_similarity(bone_pred, bone_gt, dim=-1)  # [B]
        angular_error = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))  # 弧度
        bone_errors.append(angular_error)

    if bone_errors:
        dir_error = torch.cat(bone_errors).mean()
    else:
        dir_error = torch.tensor(0.0)

    # 归一化组合
    npp = 1.0 / (1.0 + alpha * pos_error + (1 - alpha) * dir_error)
    return npp.item()


def compute_mpe(pred, gt):
    """计算每个关节的平均位置误差"""
    joint_errors = torch.norm(pred - gt, dim=-1)  # [B, J]
    mpe_per_joint = joint_errors.mean(dim=0) * 1000  # 转换为mm
    return mpe_per_joint


def compute_fscore(pred, gt, threshold=0.1):
    """
    基于距离阈值的F1分数
    将姿态估计视为分类问题：正确预测的关节 vs 错误预测的关节
    """
    distances = torch.norm(pred - gt, dim=-1)  # [B, J]

    # True Positive: 距离小于阈值的关节
    tp = (distances < threshold).sum().float()

    # False Positive: 距离大于等于阈值的预测关节
    fp = (distances >= threshold).sum().float()

    # False Negative: 这里与TP定义相同，因为每个关节都必须预测
    fn = fp.clone()  # 在这种定义下，FP = FN

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    return fscore.item()


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
    var1 = torch.sum(X1 ** 2, dim=(1, 2), keepdim=True)

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