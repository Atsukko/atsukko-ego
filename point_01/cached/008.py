from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple
from framevision import geometry as geo


class EnhancedLearnableGraphConv(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_joints: int, bias: bool = True,
                 hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # 多尺度特征变换
        self.W1 = nn.Parameter(torch.zeros(size=(in_features, hidden_dim), dtype=torch.float))
        self.W2 = nn.Parameter(torch.zeros(size=(hidden_dim, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        # 多头邻接矩阵 - 学习不同的关节关系模式
        self.adj_heads = nn.ParameterList([
            nn.Parameter(torch.eye(num_joints, dtype=torch.float) * 0.8 +
                        torch.randn(num_joints, num_joints) * 0.2 / num_joints)
            for _ in range(num_heads)
        ])

        # 注意力机制来动态调整邻接矩阵权重
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 残差连接和归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # 可学习的门控机制
        self.gate_alpha = nn.Parameter(torch.tensor(0.5))
        self.gate_beta = nn.Parameter(torch.tensor(0.5))

        if bias:
            self.bias1 = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float))
            self.bias2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

        # 物理启发的邻接矩阵初始化
        self.register_buffer('physical_mask', self.create_physical_mask())

    def create_physical_mask(self):
        """创建基于人体结构的物理约束掩码（可选）"""
        mask = torch.ones(self.num_joints, self.num_joints)
        # 这里可以添加基于人体解剖学的先验知识
        # 例如：相邻关节的连接强度应该更高
        return mask

    def symmetric_softmax(self, adj):
        """对称的softmax归一化"""
        adj = (adj + adj.transpose(-1, -2)) / 2  # 强制对称
        return F.softmax(adj, dim=-1)

    def forward(self, input: Tensor) -> Tensor:
        B, T, V, J, C = input.shape

        # 重塑为图卷积格式 (B*T*V, J, C)
        x = input.reshape(B * T * V, J, C)

        # 第一层变换
        x_transformed = torch.matmul(x, self.W1)
        if self.bias1 is not None:
            x_transformed = x_transformed + self.bias1

        # 多头图卷积
        head_outputs = []
        for adj_head in self.adj_heads:
            adj_norm = self.symmetric_softmax(adj_head)
            # 应用物理约束掩码
            adj_norm = adj_norm * self.physical_mask
            head_output = torch.matmul(adj_norm, x_transformed)
            head_outputs.append(head_output)

        # 多头融合
        if len(head_outputs) > 1:
            x_graph = torch.stack(head_outputs).mean(dim=0)
        else:
            x_graph = head_outputs[0]

        # 残差连接和归一化
        x_residual = x_transformed + self.gate_alpha * x_graph
        x_norm = self.norm1(x_residual)
        x_activated = F.gelu(x_norm)
        x_dropout = self.dropout(x_activated)

        # 注意力机制增强特征
        x_attn, _ = self.attention(x_dropout, x_dropout, x_dropout)
        x_enhanced = x_dropout + self.gate_beta * x_attn

        # 第二层变换
        x_output = torch.matmul(x_enhanced, self.W2)
        if self.bias2 is not None:
            x_output = x_output + self.bias2

        # 最终归一化
        x_output = self.norm2(x_output)

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
            use_graph_conv: bool = True,
            graph_hidden_dim: int = 64,
            graph_num_heads: int = 4
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_conv = use_graph_conv
        self.time_steps = time_steps

        in_dim = num_views * num_keypoints * 3

        # 增强的图卷积层
        if use_graph_conv:
            self.graph_conv = EnhancedLearnableGraphConv(
                in_features=3,
                out_features=3,
                num_joints=num_keypoints,
                hidden_dim=graph_hidden_dim,
                num_heads=graph_num_heads,
                dropout=dropout
            )
            self.graph_norm = nn.LayerNorm(3)
            # 可学习的残差权重
            self.residual_weight = nn.Parameter(torch.tensor(0.1))

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
        self.output_norm = nn.LayerNorm(in_dim // num_views)

    def forward(self, joints_3D: Tensor) -> Tensor:
        B, T, V, J, _ = joints_3D.shape

        # 应用增强的图卷积
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            # 使用可学习权重的残差连接
            joints_3D = original_joints + self.residual_weight * joints_3D

        joints_3D_fl_flat = self.flatten(joints_3D)

        x = self.embedding(joints_3D_fl_flat)

        x = self.positional_encoding(x) + x

        x = self.transformer_encoder(x)

        # 输出投影
        x = self.output_layer(x)
        x = self.output_norm(x)

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
            graph_hidden_dim: int = 64,
            graph_num_heads: int = 4,
            **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}

        # 使用增强的时空Transformer
        self.transformer = EnhancedSpatioTemporalTransformer(
            num_keypoints=num_keypoints,
            num_views=num_views,
            time_steps=time_steps,
            use_graph_conv=use_graph_conv,
            graph_hidden_dim=graph_hidden_dim,
            graph_num_heads=graph_num_heads,
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