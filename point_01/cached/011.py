from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple

from framevision import geometry as geo


class LearnableGraphConv(nn.Module):
    """修复后的可学习图卷积层"""

    def __init__(self, in_features: int, out_features: int, num_joints: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints

        # 简化权重设计，避免过度参数化
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 可学习的邻接矩阵，使用更合理的初始化
        self.adj = nn.Parameter(torch.eye(num_joints, dtype=torch.float) * 0.9 +
                                torch.ones(num_joints, num_joints) * 0.1 / num_joints)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        B, T, V, J, C = input.shape

        # 重塑为图卷积格式 (B*T*V, J, C)
        x = input.reshape(B * T * V, J, C)

        # 应用线性变换
        x_transformed = torch.matmul(x, self.W)  # (B*T*V, J, out_features)

        # 对称化邻接矩阵
        adj = (self.adj + self.adj.T) / 2
        adj = F.softmax(adj, dim=-1)  # 归一化

        # 图卷积操作
        x_output = torch.matmul(adj, x_transformed)  # (B*T*V, J, out_features)

        if self.bias is not None:
            x_output = x_output + self.bias

        return x_output.reshape(B, T, V, J, self.out_features)

class PositionalEncoding(nn.Module):
    """修复的位置编码，确保维度匹配"""

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
        # 确保位置编码的维度与输入匹配
        if T <= self.max_len:
            return self.pos_enc[:, :T]
        else:
            # 如果输入序列更长，进行插值
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

        # 图卷积层（可选）
        if use_graph_conv:
            # 图卷积保持3维输出，不改变坐标维度
            self.graph_conv = LearnableGraphConv(3, 3, num_keypoints)
            # 在正确维度上应用归一化
            self.graph_norm = nn.LayerNorm(3)  # 在坐标维度归一化

        # 保持原始STF的嵌入层
        self.embedding = nn.Linear(in_dim, embed_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(time_steps, embed_dim)

        # 保持原始STF的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层 - 保持原始设计
        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)

    def forward(self, joints_3D: Tensor) -> Tensor:
        B, T, V, J, _ = joints_3D.shape

        # 可选：应用图卷积增强空间关系
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            # 残差连接保持原始信息
            joints_3D = original_joints + 0.1 * joints_3D  # 小权重融合

        joints_3D_fl_flat = self.flatten(joints_3D)

        x = self.embedding(joints_3D_fl_flat)

        x = self.positional_encoding(x) + x

        x = self.transformer_encoder(x)

        # 输出投影
        x = self.output_layer(x)

        return self.unflatten(x)

    def flatten(self, joints_3D: Tensor):
        B, T, V, J, _ = joints_3D.shape
        self._out_shape = (B, T, J, 3)
        return joints_3D.view(B, T, V * J * 3)

    def unflatten(self, joints_3D: Tensor):
        return joints_3D.view(self._out_shape)


class STF(nn.Module):

    def __init__(
            self,
            num_keypoints: int,
            time_steps: int,
            num_views: int = 2,
            undersampling_factor: int = 1,
            transform_kwargs: Optional[dict] = None,
            use_graph_conv: bool = True,
            graph_conv_weight: float = 0.1,  # 控制图卷积影响程度
            **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.graph_conv_weight = graph_conv_weight

        # 使用修复的增强时空Transformer
        self.transformer = EnhancedSpatioTemporalTransformer(
            num_keypoints=num_keypoints,
            num_views=num_views,
            time_steps=time_steps,
            use_graph_conv=use_graph_conv,
            **kwargs
        )

    def forward(self, joints_3D_cc, left2middle, right2middle, middle2world, **kwargs):
        # 保持原有的坐标变换流程
        B, T, V, J, _ = joints_3D_cc.shape

        # 计算变换矩阵（保持原有逻辑）
        cams2floor, floor2world = self.compute_transformations(
            left2middle, right2middle, middle2world
        )

        # 坐标变换
        joints_3D = geo.rototranslate(joints_3D_cc, cams2floor)

        # 通过增强的Transformer
        joints_3D_fl = self.transformer(joints_3D)
        joints_3D_wr = geo.rototranslate(joints_3D_fl, floor2world)

        last_pred_last_step = joints_3D_wr[:, -1:]
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
        middle2floor_last = geo.compute_relpose_to_floor(middle2world_last,
                                                         **self.transform_kwargs)  # Shape: (B, 1, 4, 4)
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