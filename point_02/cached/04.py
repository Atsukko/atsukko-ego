from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple

from framevision import geometry as geo

def add_joint_noise(joints, sigma=0.01):
    noise = torch.randn_like(joints) * sigma
    return joints + noise

def joint_dropout(joints, drop_prob=0.1):
    B, T, V, J, C = joints.shape
    mask = (torch.rand(B, T, V, J, 1, device=joints.device) > drop_prob).float()
    return joints * mask

def temporal_jitter(joints, jitter_std=0.005):
    noise = torch.randn_like(joints) * jitter_std
    return joints + noise


def get_skeleton_adj(num_joints: int):
    adj = torch.zeros(num_joints, num_joints)
    # 按照你提供的列表顺序：
    # 0:Neck, 1:L_Arm, 2:L_ForeArm, 3:L_Hand,
    # 4:R_Arm, 5:R_ForeArm, 6:R_Hand,
    # 7:L_UpLeg, 8:L_Leg, 9:L_Foot, 10:L_Toe,
    # 11:R_UpLeg, 12:R_Leg, 13:R_Foot, 14:R_Toe

    edges = [
        (0, 1), (1, 2), (2, 3),  # 左臂链
        (0, 4), (4, 5), (5, 6),  # 右臂链
        (0, 7), (7, 8), (8, 9), (9, 10),  # 左腿链
        (0, 11), (11, 12), (12, 13), (13, 14)  # 右腿链
    ]

    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    for i in range(num_joints):
         adj[i, i] = 1  # 自环
    return adj


class LearnableGraphConv(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_joints: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        adj1 = get_skeleton_adj(num_joints)
        self.register_buffer("adj1", adj1)
        self.adj2 = nn.Parameter(torch.ones(num_joints, num_joints) / num_joints)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        ##################参数初始化函数########################
        self.reset_parameters()

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.W.size(1))
        # size包括(in_features, out_features)，size(1)应该是指out_features
        # stdv=1/根号(out_features)
        self.W.data.uniform_(-stdv, stdv)
        # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        if self.bias is not None:  # 变量是否不是None
            self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    def forward(self, input: Tensor) -> Tensor:
        B, T, V, J, C = input.shape

        # 重塑为图卷积格式 (B*T*V, J, C)
        x = input.reshape(B * T * V, J, C)

        # 应用线性变换
        x_transformed = torch.matmul(x, self.W)  # (B*T*V, J, out_features)

        w = torch.sigmoid(self.alpha)
        A = (1 - w) * self.adj1 + w * self.adj2

        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        # 图卷积操作
        x_output = torch.matmul(A, x_transformed)  # (B*T*V, J, out_features)

        if self.bias is not None:
            x_output = x_output + self.bias

        return x_output.reshape(B, T, V, J, self.out_features)

class ViewJointReliability(nn.Module):
    """
    Reliability modeling for each (view, joint) observation.

    - Structure-aware: uses relative joint geometry
    - Temporally consistent: smooths reliability (not coordinates)
    - Lightweight and plug-and-play
    """

    def __init__(
        self,
        num_joints: int,
        num_views: int,
        hidden_dim: int = 32,
        temporal_kernel: int = 3
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_views = num_views

        # ---------- Structure-aware reliability encoder ----------
        # Input: [x, y, z, dx, dy, dz]
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # ---------- Temporal consistency on reliability ----------
        # Depthwise temporal conv: each joint independently
        padding = temporal_kernel // 2
        self.temporal_smoother = nn.Conv1d(
            in_channels=num_joints,
            out_channels=num_joints,
            kernel_size=temporal_kernel,
            padding=padding,
            groups=num_joints,
            bias=False
        )

        # Initialize as identity-like smoothing
        nn.init.constant_(self.temporal_smoother.weight, 1.0 / temporal_kernel)

    def forward(self, joints):
        """
        Args:
            joints: Tensor of shape (B, T, V, J, 3)

        Returns:
            reliability: Tensor of shape (B, T, V, J, 1), range (0,1)
        """
        B, T, V, J, C = joints.shape
        assert J == self.num_joints
        assert V == self.num_views

        # --------------------------------------------------
        # 1. Structure-aware feature construction
        # --------------------------------------------------
        # Reference joint position across views (mean as proxy)
        joint_ref = joints.mean(dim=2, keepdim=True)   # (B,T,1,J,3)

        # Relative offset (captures geometric inconsistency)
        joint_delta = joints - joint_ref               # (B,T,V,J,3)

        # Feature: absolute + relative
        feat = torch.cat([joints, joint_delta], dim=-1)  # (B,T,V,J,6)

        # --------------------------------------------------
        # 2. Joint–View reliability prediction
        # --------------------------------------------------
        feat = feat.view(B * T * V * J, 6)
        w = torch.sigmoid(self.mlp(feat))
        w = w.view(B, T, V, J, 1)

        # --------------------------------------------------
        # 3. Temporal consistency on reliability
        # --------------------------------------------------
        # Reshape: (B, J, T, V)
        w_t = w.permute(0, 3, 1, 2, 4).squeeze(-1)

        # Merge batch and view for temporal conv
        w_t = w_t.reshape(B * V, J, T)

        # Temporal smoothing
        w_t = self.temporal_smoother(w_t)

        # Restore shape
        w_t = w_t.view(B, V, J, T).permute(0, 3, 1, 2)
        w = w_t.unsqueeze(-1)  # (B,T,V,J,1)

        return w

def reliability_weighted_fusion(joints, weights):
    # joints: (B, T, V, J, 3)
    # weights: (B, T, V, J, 1)

    weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
    fused = (joints * weights).sum(dim=2)  # sum over views
    return fused  # (B, T, J, 3)



class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        time_steps: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_graph_conv: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = num_keypoints * 3

        seq_len, in_dim = time_steps, num_views * num_keypoints * 3

        self.use_graph_conv = use_graph_conv

        # 图卷积层（可选）
        if use_graph_conv:
            # 图卷积保持3维输出，不改变坐标维度
            self.graph_conv = LearnableGraphConv(3, 3, num_keypoints)
            # 在正确维度上应用归一化
            self.graph_norm = nn.LayerNorm(3)  # 在坐标维度归一化

        self.embedding = nn.Linear(self.input_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, self.input_dim)

        # Positional encoding to retain temporal and joint position information
        self.positional_encoding = PositionalEncoding(max_len=seq_len, embed_dim=embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 添加可学习的残差权重
        self.graph_res_weight = nn.Parameter(torch.tensor(0.1))

        self.temporal_mix = nn.Conv1d(
            in_channels=num_keypoints * 3,
            out_channels=num_keypoints * 3,
            kernel_size=3,
            padding=1,
            groups=num_keypoints,  # 每个 joint 一组
            bias=False
        )
        self.joint_gate = nn.Parameter(0.88 * torch.ones(1, in_dim, 1))

    def forward(self, joints_3D: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            joints_3D: Input tensor of shape (B, T, V, J, 3).

        Returns:
            Output tensor of shape (B, T, J, 3)
        """

        B, T, V, J, C = joints_3D.shape

        # 应用图卷积增强空间关系
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            joints_3D = original_joints + self.graph_res_weight * joints_3D

        # joints_3D: (B, T, V=1, J, 3)
        x = joints_3D.squeeze(2)  # (B, T, J, 3)
        x = x.permute(0, 2, 3, 1)  # (B, J, 3, T)
        x = x.reshape(B, J * 3, T)  # (B, J*3, T)

        x_tm = self.temporal_mix(x)

        x = x + x_tm  # residual
        x = x.view(B, J, 3, T)
        x = x.permute(0, 3, 1, 2)  # (B, T, J, 3)
        joints_3D = x.unsqueeze(2)  # (B, T, 1, J, 3)

        joints_3D_fl_flat = self.flatten(joints_3D)

        # Apply embedding layer
        x = self.embedding(joints_3D_fl_flat)  # Shape: (B, T, embed_dim)

        x = self.positional_encoding(x) + x  # Add positional encoding

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (B, T, embed_dim)

        # Project back to original input dimension for all time steps
        x = self.output_layer(x)  # Shape: (B, T, C)

        return self.unflatten(x)

    def flatten(self, joints_3D: Tensor):
        B, T, V, J, _ = joints_3D.shape
        self._out_shape = (B, T, J, 3)
        return joints_3D.view(B, T, V * J * 3)

    def unflatten(self, joints_3D: Tensor):
        return joints_3D.view(self._out_shape)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int, scale: float = 10000.0, inverted: bool = True):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1) if not inverted else torch.arange(max_len - 1, -1, -1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(scale) / embed_dim))

        pos_enc = torch.zeros(max_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        return self.pos_enc[:, -length:]



class STF(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        time_steps: int,
        num_views: int = 2,
        undersampling_factor: int = 1,
        transform_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.transformer = SpatioTemporalTransformer(num_keypoints, num_views, time_steps, **kwargs)
        self.reliability_net = ViewJointReliability(
            num_joints=num_keypoints,
            num_views=num_views
        )

    def forward(
        self,
        joints_3D_cc: Float[Tensor, "B T V J 3"],
        left2middle: Float[Tensor, "B 4 4"],
        right2middle: Float[Tensor, "B 4 4"],
        middle2world: Float[Tensor, "B T 4 4"],
        **kwargs,
    ):
        """
        Forward pass to predict 3D keypoints from a history of 3D keypoints in camera coordinates.

        Args:
            joints_3D_cc: Predictions of 3D keypoints in camera coordinates for the previous T frames.
            middle2world: VR pose tracking over the past T frames in world coordinates. This is the M frame of reference in the paper.
            left2middle: Transformation matrix from left camera to the middle/VR frame of reference. This is coming from calibration.
            right2middle: Transformation matrix from right camera to the middle/VR frame of reference. This is coming from calibration.
        """
        if self.training:
            joints_3D_cc = add_joint_noise(joints_3D_cc, sigma=0.03)
            joints_3D_cc = joint_dropout(joints_3D_cc, drop_prob=0.3)
            joints_3D_cc = temporal_jitter(joints_3D_cc, jitter_std=0.01)

        B, T, V, J, _ = joints_3D_cc.shape

        cams2floor, floor2world = self.compute_transformations(left2middle, right2middle, middle2world)
        joints_3D = geo.rototranslate(joints_3D_cc, cams2floor)

        # joints_3D: (B, T, V, J, 3)
        reliability = self.reliability_net(joints_3D)  # (B, T, V, J, 1)
        joints_3D_fused = reliability_weighted_fusion(joints_3D, reliability)

        # transformer 现在吃的是“可靠融合后的关节”
        joints_3D_fl = self.transformer(joints_3D_fused.unsqueeze(2))

        joints_3D_wr = geo.rototranslate(joints_3D_fl, floor2world)

        last_pred_last_step = joints_3D_wr[:, -1:]  # Shape: (B, 1, J, 3)
        return dict(joints_3D=last_pred_last_step, all_joints_3D=joints_3D_wr)

    @torch.autocast("cuda", enabled=False)
    def compute_transformations(self, left2middle, right2middle, middle2world):
        """
        Args:
            left2middle: Transformation matrix from left to middle camera frame. Shape: (B, 4, 4).
            right2middle: Transformation matrix from right to middle camera frame. Shape: (B, 4, 4).
            middle2world: Transformation matrix from middle to world frame. Shape: (B, T, 4, 4).

        Returns:
            cams2floor_last: Transformation matrix from cameras to the last floor frame. Shape: (B, T, 2, 4, 4).
            floor_last2world: Transformation matrix from the last floor frame to the world frame. Shape: (B, T, 4, 4).
        """

        # Computing the transformations from the cameras to the middle frame
        cams2middle = torch.stack([left2middle, right2middle], dim=1)  # Shape: (B, 2, 4, 4)

        # Compute the transformation from world coordinate to the last floor frame
        middle2world_last = middle2world[:, -1].unsqueeze(1)  # Shape: (B, 1, 4, 4)
        middle2floor_last = geo.compute_relpose_to_floor(middle2world_last, **self.transform_kwargs)  # Shape: (B, 1, 4, 4)
        world2floor_last = middle2floor_last @ geo.invert_SE3(middle2world_last)  # Shape: (B, 1, 4, 4)
        floor_last2world = geo.invert_SE3(world2floor_last)  # Shape: (B, 1, 4, 4)

        # Unsqueeze approriate dimension to make sure they match
        cams2middle = cams2middle.unsqueeze(1)  # Shape: (B, 1, 2, 4, 4)
        middle2world = middle2world.unsqueeze(2)  # Shape: (B, T, 1, 4, 4)

        # Compute the transformation from the cameras to world coordinates
        cams2world = middle2world @ cams2middle  # Shape: (B, T, 2, 4, 4)

        # Compute the transformation from the cameras to the last floor frame
        world2floor_last = world2floor_last.unsqueeze(2)  # Shape: (B, 1, 1, 4, 4)
        cams2floor_last = world2floor_last @ cams2world  # Shape: (B, T, 2, 4, 4)

        return cams2floor_last, floor_last2world