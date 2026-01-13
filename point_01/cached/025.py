from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple

from framevision import geometry as geo


class SpatioTemporalAttention(nn.Module):
    """
    时空注意力模块，主要增强时空关系的建模能力
    """

    def __init__(self, num_joints: int, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        head_dim = embed_dim // num_heads

        self.scale = head_dim ** -1

        # QKV投影
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # 可学习的时空位置编码
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 简化的FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        # 可学习的残差权重
        self.res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, T, J, C)
        Returns:
            增强后的特征，形状为 (B, T, J, C)
        """
        B, T, J, C = x.shape
        residual = x

        # 应用层归一化
        x = self.norm1(x)

        # 添加时空位置编码
        temporal_embed = self.temporal_pos_embed.unsqueeze(2)  # (1, 1, 1, C)
        spatial_embed = self.spatial_pos_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, J, C)
        x = x + temporal_embed + spatial_embed

        # 重塑为注意力计算格式
        x_flat = x.reshape(B * T, J, C)

        # QKV投影
        qkv = self.qkv_proj(x_flat).reshape(B * T, J, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, heads, J, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 注意力加权
        x_attn = (attn @ v).transpose(1, 2).reshape(B * T, J, C)
        x_attn = self.output_proj(x_attn)
        x_attn = x_attn.reshape(B, T, J, C)

        # 残差连接
        x = residual + self.res_weight * x_attn

        # FFN部分
        residual_ffn = x
        x = self.norm2(x)
        x_ffn = self.ffn(x)
        x = residual_ffn + self.res_weight * x_ffn

        return x


class LearnableGraphConv(nn.Module):
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


class PositionalEncoding(nn.Module):
    # 保持原有的位置编码代码不变
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
            use_graph_conv: bool = True,
            use_st_attention: bool = True  # 新增：是否使用时空注意力
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_conv = use_graph_conv
        self.use_st_attention = use_st_attention
        self.time_steps = time_steps
        self.num_keypoints = num_keypoints
        self.num_views = num_views

        in_dim = num_views * num_keypoints * 3

        # 图卷积层（可选）
        if use_graph_conv:
            self.graph_conv = LearnableGraphConv(3, 3, num_keypoints)
            self.graph_norm = nn.LayerNorm(3)
            self.graph_res_weight = nn.Parameter(torch.tensor(0.1))

        # 时空注意力模块（可选）
        if use_st_attention:
            # 注意：这里使用专门的嵌入维度来处理关节特征
            self.joint_embed_dim = embed_dim // 2  # 使用一半的嵌入维度
            self.joint_embedding = nn.Linear(3, self.joint_embed_dim)  # 将3D坐标映射到更高维度

            self.st_attention = SpatioTemporalAttention(
                num_joints=num_keypoints * num_views,  # 考虑所有视图的关节
                embed_dim=self.joint_embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.st_attention_res_weight = nn.Parameter(torch.tensor(0.1))

            # 将关节特征映射回原始维度
            self.joint_projection = nn.Linear(self.joint_embed_dim, 3)

        # 保持原始STF的嵌入层
        self.embedding = nn.Linear(in_dim, embed_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(time_steps, embed_dim)

        # 保持原始STF的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(embed_dim, in_dim // num_views)

    def forward(self, joints_3D: Tensor) -> Tensor:
        B, T, V, J, _ = joints_3D.shape

        # 可选：应用图卷积增强空间关系
        if self.use_graph_conv:
            original_joints = joints_3D
            joints_3D = self.graph_conv(joints_3D)
            joints_3D = self.graph_norm(joints_3D)
            joints_3D = original_joints + self.graph_res_weight * joints_3D

        # 新增：应用时空注意力（在图卷积之后，展平之前）
        if self.use_st_attention:
            original_joints_st = joints_3D

            # 将3D坐标映射到更高维度
            joints_3D_embed = self.joint_embedding(joints_3D)  # (B, T, V, J, joint_embed_dim)

            # 合并视图和关节维度，形成 (B, T, V*J, joint_embed_dim)
            joints_3D_reshaped = joints_3D_embed.reshape(B, T, V * J, self.joint_embed_dim)

            # 应用时空注意力
            joints_3D_attn = self.st_attention(joints_3D_reshaped)  # (B, T, V*J, joint_embed_dim)

            # 映射回3D坐标空间
            joints_3D_attn_3d = self.joint_projection(joints_3D_attn)  # (B, T, V*J, 3)

            # 恢复原始形状
            joints_3D_attn_3d = joints_3D_attn_3d.reshape(B, T, V, J, 3)

            # 残差连接
            joints_3D = original_joints_st + self.st_attention_res_weight * joints_3D_attn_3d

        # 扁平化并嵌入
        joints_3D_fl_flat = self.flatten(joints_3D)
        x = self.embedding(joints_3D_fl_flat)
        x = self.positional_encoding(x) + x

        # Transformer编码器
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
            use_st_attention: bool = True,  # 新增参数
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
            use_st_attention=use_st_attention,  # 传递新参数
            **kwargs
        )

    def forward(self, joints_3D_cc, left2middle, right2middle, middle2world, **kwargs):
        # 保持原有的坐标变换流程
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
        # 保持原有的变换计算代码不变
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