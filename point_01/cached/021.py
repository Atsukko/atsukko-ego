from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple

from framevision import geometry as geo


class MultiScaleTemporalAttention(nn.Module):
    """多尺度时间注意力模块"""

    def __init__(self, num_joints: int, feature_dim: int, num_heads: int = 4, scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.scales = scales

        # 确保num_heads能够整除feature_dim
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads

        # 多尺度卷积
        self.conv_layers = nn.ModuleList()
        for scale in scales:
            if scale == 1:
                conv = nn.Identity()
            else:
                kernel_size = scale * 2 - 1
                padding = (kernel_size - 1) // 2
                conv = nn.Conv1d(
                    feature_dim, feature_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=feature_dim,  # 深度可分离卷积减少参数量
                    bias=False
                )
                # 初始化卷积权重
                nn.init.normal_(conv.weight, mean=0, std=0.01)
            self.conv_layers.append(conv)

        # 跨尺度注意力机制
        self.cross_scale_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True, dropout=0.1
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(len(scales) * feature_dim, len(scales)),
            nn.Softmax(dim=-1)
        )

        # 层归一化
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        """输入: (B, T, V, J, C), 输出: (B, T, V, J, C)"""
        B, T, V, J, C = x.shape

        # 重塑为时间序列格式 (B*V*J, T, C)
        x_reshaped = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, V, J, T, C)
        x_reshaped = x_reshaped.view(B * V * J, T, C)  # (B*V*J, T, C)

        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.conv_layers:
            if isinstance(conv, nn.Identity):
                # 恒等映射
                conv_out = x_reshaped
            else:
                # 转置进行1D卷积 (B*V*J, T, C) -> (B*V*J, C, T)
                x_conv = x_reshaped.transpose(1, 2)
                x_conv = conv(x_conv)  # (B*V*J, C, T)
                x_conv = x_conv.transpose(1, 2)  # (B*V*J, T, C)
                conv_out = x_conv
            multi_scale_features.append(conv_out)

        # 跨尺度注意力
        if len(multi_scale_features) > 1:
            scale_features = torch.stack(multi_scale_features, dim=1)  # (B*V*J, num_scales, T, C)
            scale_features_flat = scale_features.view(B * V * J, len(self.scales) * T, C)

            # 自注意力增强
            attended_features, _ = self.cross_scale_attention(
                scale_features_flat, scale_features_flat, scale_features_flat
            )
            attended_features = attended_features.view(B * V * J, len(self.scales), T, C)

            # 门控融合
            gate_input = attended_features.mean(dim=2)  # (B*V*J, num_scales, C)
            gate_input = gate_input.view(B * V * J, len(self.scales) * C)
            gate_weights = self.gate(gate_input).unsqueeze(-1).unsqueeze(-1)  # (B*V*J, num_scales, 1, 1)

            # 加权融合
            fused_features = (attended_features * gate_weights).sum(dim=1)  # (B*V*J, T, C)
        else:
            # 如果只有一个尺度，直接使用
            fused_features = multi_scale_features[0]

        # 残差连接和归一化
        # output = self.norm(fused_features + x_reshaped)
        output = fused_features + x_reshaped

        # 恢复原始形状
        output = output.view(B, V, J, T, C).permute(0, 3, 1, 2, 4).contiguous()  # (B, T, V, J, C)

        return output


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
            use_graph_conv: bool = True,
            use_multi_scale_attn: bool = True  # 新增：控制多尺度注意力模块
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_conv = use_graph_conv
        self.use_multi_scale_attn = use_multi_scale_attn
        self.time_steps = time_steps

        in_dim = num_views * num_keypoints * 3

        # 图卷积层（可选）
        if use_graph_conv:
            # 图卷积保持3维输出，不改变坐标维度
            self.graph_conv = LearnableGraphConv(3, 3, num_keypoints)
            # 在正确维度上应用归一化
            self.graph_norm = nn.LayerNorm(3)  # 在坐标维度归一化

        # 多尺度时间注意力模块（可选）
        if use_multi_scale_attn:
            # 确保feature_dim能被num_heads整除
            temporal_feature_dim = 4  # 使用4而不是3，因为4能被常用的num_heads整除
            self.multi_scale_attn = MultiScaleTemporalAttention(
                num_joints=num_keypoints,
                feature_dim=temporal_feature_dim,  # 使用新的特征维度
                num_heads=2,  # 确保能被feature_dim整除
                scales=(1, 2, 4)  # 多尺度时间窗口
            )
            # 输入输出投影层，将3维坐标映射到temporal_feature_dim维
            self.temporal_input_proj = nn.Linear(3, temporal_feature_dim)
            self.temporal_output_proj = nn.Linear(temporal_feature_dim, 3)
            self.temporal_norm = nn.LayerNorm(3)

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

        # 添加可学习的残差权重
        self.graph_res_weight = nn.Parameter(torch.tensor(0.1))
        if use_multi_scale_attn:
            self.temporal_res_weight = nn.Parameter(torch.tensor(0.1))

        # 图卷积和时间注意力的交互权重
        if use_graph_conv and use_multi_scale_attn:
            self.interaction_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, joints_3D: Tensor) -> Tensor:
        B, T, V, J, _ = joints_3D.shape
        original_joints = joints_3D

        # 可选：应用图卷积增强空间关系
        if self.use_graph_conv:
            graph_out = self.graph_conv(joints_3D)
            graph_out = self.graph_norm(graph_out)
            # 残差连接保持原始信息
            joints_3D = original_joints + self.graph_res_weight * graph_out

        # 可选：应用多尺度时间注意力
        if self.use_multi_scale_attn:
            # 将输入投影到更高的维度
            temporal_input = self.temporal_input_proj(joints_3D)
            temporal_out = self.multi_scale_attn(temporal_input)
            # 投影回原始维度
            temporal_out = self.temporal_output_proj(temporal_out)
            temporal_out = self.temporal_norm(temporal_out)

            # 如果同时使用图卷积和时间注意力，进行交互增强
            if self.use_graph_conv:
                # 图卷积输出和时间注意力输出的交互
                interactive_out = graph_out * temporal_out
                joints_3D = joints_3D + self.interaction_weight * interactive_out
            else:
                joints_3D = joints_3D + self.temporal_res_weight * temporal_out

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
            use_multi_scale_attn: bool = True,
            graph_conv_weight: float = 0.1,
            # 新增简单改进
            use_deep_supervision: bool = True,  # 深度监督
            dropout_rate: float = 0.2,  # 增加dropout
            **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.graph_conv_weight = graph_conv_weight
        self.use_deep_supervision = use_deep_supervision

        # 从kwargs中移除dropout，避免重复传递
        transformer_kwargs = kwargs.copy()
        if 'dropout' in transformer_kwargs:
            del transformer_kwargs['dropout']

        # 使用修复的增强时空Transformer
        self.transformer = EnhancedSpatioTemporalTransformer(
            num_keypoints=num_keypoints,
            num_views=num_views,
            time_steps=time_steps,
            use_graph_conv=use_graph_conv,
            use_multi_scale_attn=use_multi_scale_attn,
            dropout=dropout_rate,  # 明确传递dropout_rate
            **transformer_kwargs  # 传递其他参数但不包含dropout
        )

        # 深度监督：中间层预测
        if use_deep_supervision:
            embed_dim = kwargs.get('embed_dim', 512)
            self.auxiliary_output = nn.Linear(embed_dim, num_views * num_keypoints * 3 // num_views)

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

        output_dict = dict(joints_3D=last_pred_last_step, all_joints_3D=joints_3D_wr)

        # 深度监督：添加中间监督信号
        if self.use_deep_supervision and self.training:
            # 这里需要修改，因为joints_3D_fl已经是最终输出
            # 我们需要在transformer内部获取中间特征
            # 暂时注释掉，需要修改transformer结构
            pass

        return output_dict

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