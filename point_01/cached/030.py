from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple

from framevision import geometry as geo


class FixedGraphConv(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_joints: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints

        # 权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 固定邻接矩阵 - 基于人体结构先验
        self.register_buffer("adj", self._build_fixed_adjacency_matrix())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def _build_fixed_adjacency_matrix(self) -> Tensor:
        """基于人体结构先验构建固定邻接矩阵"""
        adj = torch.zeros(self.num_joints, self.num_joints, dtype=torch.float)

        # 定义人体关节连接关系
        connections = [
            # 躯干和四肢的连接
            (0, 1), (0, 4), (0, 7), (0, 11),  # Neck连接到四肢

            # 左臂连接链
            (1, 2), (2, 3),  # Neck -> LeftArm -> LeftForeArm -> LeftHand

            # 右臂连接链
            (4, 5), (5, 6),  # Neck -> RightArm -> RightForeArm -> RightHand

            # 左腿连接链
            (7, 8), (8, 9), (9, 10),  # Neck -> LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToeBase

            # 右腿连接链
            (11, 12), (12, 13), (13, 14),  # Neck -> RightUpLeg -> RightLeg -> RightFoot -> RightToeBase

            # 对称连接（可选，增强左右对称性）
            (1, 4), (7, 11),  # 左右对称关节连接
        ]

        # 填充邻接矩阵（无向图，所以双向连接）
        for i, j in connections:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        # 添加自连接
        for i in range(self.num_joints):
            adj[i, i] = 1.0

        return adj

    def forward(self, input: Tensor) -> Tensor:
        B, T, V, J, C = input.shape

        # 重塑为图卷积格式 (B*T*V, J, C)
        x = input.reshape(B * T * V, J, C)

        # 应用线性变换
        x_transformed = torch.matmul(x, self.W)  # (B*T*V, J, out_features)

        # 对固定邻接矩阵进行归一化（保持对称性）
        adj = self.adj.clone()
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_normalized = degree_inv_sqrt * adj * degree_inv_sqrt.T

        # 图卷积操作
        x_output = torch.matmul(adj_normalized, x_transformed)  # (B*T*V, J, out_features)

        if self.bias is not None:
            x_output = x_output + self.bias

        return x_output.reshape(B, T, V, J, self.out_features)


class PositionalEncoding(nn.Module):
    # 保持不变
    def __init__(self, max_len: int, embed_dim: int, scale: float = 10000.0, inverted: bool = True):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        position = torch.arange(max_len).unsqueeze(1) if not inverted else torch.arange(max_len - 1, -1, -1).unsqueeze(
            1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(scale) / embed_dim))

        pos_enc = torch.zeros(max_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        if T <= self.max_len:
            return self.pos_enc[:, :T]
        else:
            pos_enc = F.interpolate(
                self.pos_enc.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            return pos_enc


class EnhancedSpatioTemporalTransformer(nn.Module):

    def __init__(
            self,
            num_keypoints: int,
            num_views: int,
            time_steps: int,
            embed_dim: int = 512,
            num_heads: int = 8,
            num_layers: int = 4,
            dropout: float = 0.1,
            use_graph_conv: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_conv = use_graph_conv
        self.time_steps = time_steps

        in_dim = num_views * num_keypoints * 3

        # 使用固定图卷积层替换可学习图卷积
        if use_graph_conv:
            self.graph_conv = FixedGraphConv(3, 3, num_keypoints)
            self.graph_norm = nn.LayerNorm(3)

        self.embedding = nn.Linear(in_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(time_steps, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)
        self.graph_res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, joints_3D: Tensor) -> Tensor:
        B, T, V, J, _ = joints_3D.shape

        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            joints_3D = original_joints + self.graph_res_weight * joints_3D

        joints_3D_fl_flat = self.flatten(joints_3D)
        x = self.embedding(joints_3D_fl_flat)
        x = self.positional_encoding(x) + x
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return self.unflatten(x)

    def flatten(self, joints_3D: Tensor):
        B, T, V, J, _ = joints_3D.shape
        self._out_shape = (B, T, J, 3)
        return joints_3D.view(B, T, V * J * 3)

    def unflatten(self, joints_3D: Tensor):
        return joints_3D.view(self._out_shape)


class STF(nn.Module):
    # 保持不变
    def __init__(
            self,
            num_keypoints: int,
            time_steps: int,
            num_views: int = 2,
            undersampling_factor: int = 1,
            transform_kwargs: Optional[dict] = None,
            use_graph_conv: bool = True,
            graph_conv_weight: float = 0.1,
            **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.graph_conv_weight = graph_conv_weight

        self.transformer = EnhancedSpatioTemporalTransformer(
            num_keypoints=num_keypoints,
            num_views=num_views,
            time_steps=time_steps,
            use_graph_conv=use_graph_conv,
            **kwargs
        )

    def forward(self, joints_3D_cc, left2middle, right2middle, middle2world, **kwargs):
        B, T, V, J, _ = joints_3D_cc.shape

        cams2floor, floor2world = self.compute_transformations(
            left2middle, right2middle, middle2world
        )

        joints_3D = geo.rototranslate(joints_3D_cc, cams2floor)
        joints_3D_fl = self.transformer(joints_3D)
        joints_3D_wr = geo.rototranslate(joints_3D_fl, floor2world)

        last_pred_last_step = joints_3D_wr[:, -1:]
        return dict(joints_3D=last_pred_last_step, all_joints_3D=joints_3D_wr)

    @torch.autocast("cuda", enabled=False)
    def compute_transformations(self, left2middle, right2middle, middle2world):
        cams2middle = torch.stack([left2middle, right2middle], dim=1)
        middle2world_last = middle2world[:, -1].unsqueeze(1)
        middle2floor_last = geo.compute_relpose_to_floor(middle2world_last, **self.transform_kwargs)
        world2floor_last = middle2floor_last @ geo.invert_SE3(middle2world_last)
        floor_last2world = geo.invert_SE3(world2floor_last)

        cams2middle = cams2middle.unsqueeze(1)
        middle2world = middle2world.unsqueeze(2)
        cams2world = middle2world @ cams2middle
        world2floor_last = world2floor_last.unsqueeze(2)
        cams2floor_last = world2floor_last @ cams2world

        return cams2floor_last, floor_last2world