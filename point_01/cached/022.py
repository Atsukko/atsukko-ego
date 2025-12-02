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
    """(保持不变) 修复后的可学习图卷积层"""

    def __init__(self, in_features: int, out_features: int, num_joints: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = nn.Parameter(torch.eye(num_joints, dtype=torch.float) * 0.9 +
                                torch.ones(num_joints, num_joints) * 0.1 / num_joints)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        B, T, V, J, C = input.shape
        x = input.reshape(B * T * V, J, C)
        x_transformed = torch.matmul(x, self.W)
        adj = (self.adj + self.adj.T) / 2
        adj = F.softmax(adj, dim=-1)
        x_output = torch.matmul(adj, x_transformed)
        if self.bias is not None:
            x_output = x_output + self.bias
        return x_output.reshape(B, T, V, J, self.out_features)


class TemporalConvRefiner(nn.Module):
    """
    [新增模块] 时序卷积精炼器
    在时间维度(T)上捕捉局部运动连续性，补充Transformer忽略的局部细节。
    结构：1D Conv -> GeLU -> 1D Conv (Residual)
    """

    def __init__(self, channels: int, kernel_size: int = 3, expansion: int = 2):
        super().__init__()
        self.mid_channels = channels * expansion
        # 保持时间维度长度不变 (padding)
        padding = (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, self.mid_channels),  # 使用GroupNorm适应小Batch size
            nn.GELU(),
            nn.Conv1d(self.mid_channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(1, channels)
        )

        # 可学习的残差缩放因子，初始化为0确保初始训练阶段不破坏原有特征
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        # Input: (B, T, V, J, C)
        B, T, V, J, C = x.shape

        # 1. 变形以便进行时间维度的卷积: (B*V*J, C, T)
        # 我们把每个关节点视为独立的轨迹进行平滑
        x_in = x.permute(0, 2, 3, 4, 1).reshape(B * V * J, C, T)

        # 2. 时序卷积
        out = self.net(x_in)

        # 3. 残差连接 + 恢复形状
        out = x_in + self.res_scale * out

        # Output: (B, T, V, J, C)
        return out.reshape(B, V, J, C, T).permute(0, 4, 1, 2, 3)


class PositionalEncoding(nn.Module):
    """(保持不变) 修复的位置编码"""

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
                self.pos_enc.transpose(1, 2), size=T, mode='linear', align_corners=False
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
            use_graph_conv: bool = True,
            use_temporal_refine: bool = True  # 新增控制开关
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_conv = use_graph_conv
        self.use_temporal_refine = use_temporal_refine  # 开关
        self.time_steps = time_steps

        in_dim = num_views * num_keypoints * 3

        # 1. 空间图卷积模块
        if use_graph_conv:
            self.graph_conv = LearnableGraphConv(3, 3, num_keypoints)
            self.graph_res_weight = nn.Parameter(torch.tensor(0.1))

        # 2. [新增] 局部时序精炼模块
        if use_temporal_refine:
            # 输入通道为3 (x,y,z)，卷积核大小为3帧
            self.temporal_refiner = TemporalConvRefiner(channels=3, kernel_size=3)

            # 3. Transformer 编码部分 (全局时空融合)
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(time_steps, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)

    def forward(self, joints_3D: Tensor) -> Tensor:
        # Input: (B, T, V, J, 3)

        # 先时序精炼（处理原始运动轨迹）
        if self.use_temporal_refine:
            joints_3D = self.temporal_refiner(joints_3D)

        # 再空间图卷积
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = original_joints + self.graph_res_weight * joints_3D

        # --- 阶段 3: 全局时空聚合 (Global Transformer) ---
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

    def __init__(
            self,
            num_keypoints: int,
            time_steps: int,
            num_views: int = 2,
            undersampling_factor: int = 1,
            transform_kwargs: Optional[dict] = None,
            use_graph_conv: bool = True,
            use_temporal_refine: bool = True,  # 暴露参数
            **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}

        self.transformer = EnhancedSpatioTemporalTransformer(
            num_keypoints=num_keypoints,
            num_views=num_views,
            time_steps=time_steps,
            use_graph_conv=use_graph_conv,
            use_temporal_refine=use_temporal_refine,  # 传递参数
            **kwargs
        )

    def forward(self, joints_3D_cc, left2middle, right2middle, middle2world, **kwargs):
        # ... (保持原来的坐标变换逻辑不变) ...
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
        # ... (保持原有的 compute_transformations 逻辑完全不变) ...
        # (此处省略具体实现以节省篇幅，直接复用你原有的代码即可)
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