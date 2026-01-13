import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FullyLearnableSpatioTemporalGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_joints: int,
                 time_steps: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints
        self.time_steps = time_steps

        # 权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 可学习的空间关系参数 (J, J)
        self.spatial_adj = nn.Parameter(
            torch.zeros(num_joints, num_joints, dtype=torch.float)
        )
        # 用单位矩阵初始化，保留自连接
        nn.init.eye_(self.spatial_adj.data)

        # 可学习的时间关系参数 (T-1, J, J) - 相邻帧间的关系
        self.temporal_adj = nn.Parameter(
            torch.zeros(time_steps - 1, num_joints, num_joints, dtype=torch.float)
        )
        # 初始化时间关系为单位矩阵，表示初始状态下只关注同一关节
        for t in range(time_steps - 1):
            nn.init.eye_(self.temporal_adj.data[t])

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def _construct_spatiotemporal_adjacency(self, device):
        """构建完全可学习的时空邻接矩阵"""
        # 获取当前设备上的参数
        spatial_adj = self.spatial_adj
        temporal_adj = self.temporal_adj

        # 构建完整的时空邻接矩阵 (T*J, T*J)
        full_adj = torch.zeros(self.time_steps * self.num_joints,
                               self.time_steps * self.num_joints, device=device)

        for t_i in range(self.time_steps):
            for t_j in range(self.time_steps):
                # 计算块矩阵的位置
                row_start = t_i * self.num_joints
                row_end = (t_i + 1) * self.num_joints
                col_start = t_j * self.num_joints
                col_end = (t_j + 1) * self.num_joints

                if t_i == t_j:
                    # 对角线块：空间关系（完全可学习）
                    full_adj[row_start:row_end, col_start:col_end] = spatial_adj
                elif abs(t_i - t_j) == 1:
                    # 相邻时间步块：时间关系（完全可学习）
                    time_idx = min(t_i, t_j)
                    full_adj[row_start:row_end, col_start:col_end] = temporal_adj[time_idx]
                # 注意：这里我们只建模相邻时间步的关系，非相邻时间步保持为0
                # 如果需要建模更长距离的时间关系，可以扩展这个逻辑

        return full_adj

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, V, J, C = input.shape
        assert T == self.time_steps, f"输入时间步数{T}与初始化{self.time_steps}不匹配"

        # 获取输入设备
        device = input.device

        # 重塑为 (B*V, T*J, C)
        x = input.permute(0, 2, 1, 3, 4).contiguous()  # (B, V, T, J, C)
        x = x.view(B * V, T * J, C)  # (B*V, T*J, C)

        # 应用线性变换
        x_transformed = torch.matmul(x, self.W)  # (B*V, T*J, out_features)

        # 构建时空邻接矩阵并归一化（确保在正确设备上）
        adj = self._construct_spatiotemporal_adjacency(device)  # (T*J, T*J)

        # 对邻接矩阵进行对称化和归一化
        adj = (adj + adj.T) / 2  # 确保对称性
        adj = F.softmax(adj, dim=-1)  # 行归一化

        # 图卷积操作
        x_output = torch.matmul(adj, x_transformed)  # (B*V, T*J, out_features)

        if self.bias is not None:
            x_output = x_output + self.bias

        # 重塑回原始格式 (B, T, V, J, C)
        x_output = x_output.view(B, V, T, J, self.out_features)
        x_output = x_output.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, V, J, C)

        return x_output


# 简化的PositionalEncoding类
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int, scale: float = 10000.0):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        # 创建位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(scale) / embed_dim))

        pos_enc = torch.zeros(max_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 0:
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        else:
            pos_enc[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])

        # 注册为缓冲区，但允许设备移动
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # 确保位置编码在正确设备上
        pos_enc = self.pos_enc.to(x.device)
        if T <= self.max_len:
            return pos_enc[:, :T]
        else:
            # 如果输入序列更长，进行插值
            pos_enc = F.interpolate(
                pos_enc.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            return pos_enc


# 更新EnhancedSpatioTemporalTransformer类
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

        # 使用新的完全可学习的时空图卷积层
        if use_graph_conv:
            self.graph_conv = FullyLearnableSpatioTemporalGraphConv(3, 3, num_keypoints, time_steps)
            self.graph_norm = nn.LayerNorm(3)

        # 保持原始STF的嵌入层
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(time_steps, embed_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)
        self.graph_res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, joints_3D: torch.Tensor) -> torch.Tensor:
        B, T, V, J, _ = joints_3D.shape

        # 应用时空图卷积
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            joints_3D = original_joints + self.graph_res_weight * joints_3D

        # 后续处理保持不变
        joints_3D_fl_flat = self.flatten(joints_3D)
        x = self.embedding(joints_3D_fl_flat)

        # 确保位置编码在正确设备上
        pos_enc = self.positional_encoding(x)
        x = pos_enc + x

        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return self.unflatten(x)

    def flatten(self, joints_3D: torch.Tensor):
        B, T, V, J, _ = joints_3D.shape
        self._out_shape = (B, T, J, 3)
        return joints_3D.view(B, T, V * J * 3)

    def unflatten(self, joints_3D: torch.Tensor):
        return joints_3D.view(self._out_shape)


# 保持STF类不变
class STF(nn.Module):
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

        # 使用增强的时空Transformer
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