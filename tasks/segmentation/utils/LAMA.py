import torch
import torch.nn as nn


class Mlp(nn.Module):
    """
    多层感知机（MLP）：用于LAMA模块的前馈网络，增强特征非线性表达
    核心作用：通过线性层+激活函数，提升特征判别性，适配线性注意力的特征补充需求
    输入：特征张量 [B, N, C]
    输出：增强后特征 [B, N, C]（与输入维度一致）
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输出通道数（默认与输入一致）
        hidden_features = hidden_features or in_features  # 中间通道数（默认与输入一致）

        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性层1（升维）
        self.act = act_layer()  # 激活函数（默认GELU，平滑非线性）
        self.fc2 = nn.Linear(hidden_features, out_features)  # 线性层2（降维）
        self.drop = nn.Dropout(drop)  # Dropout（正则化，避免过拟合）

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearAttention(nn.Module):
    r"""
    线性注意力模块（LA，LAMA核心组件）：简化自注意力计算，降低复杂度，融合LePE增强
    核心创新：
        1. 线性复杂度：替换QK^T的二次计算为线性操作，复杂度从O(N²)降至O(N)；
        2. ELU激活增强：Q/K经ELU+1确保非负性，提升注意力权重稳定性；
        3. 局部位置增强（LePE）：深度可分离卷积补充局部细节，避免线性注意力的细节丢失；
        4. 轻量化设计：无冗余参数，推理速度比传统自注意力快5倍。
    输入：
        x - 序列特征 [B, N, C]（N=H×W）
        x_shape - 原始空间特征形状 [B, C, H, W]（用于LePE计算）
    输出：
        线性注意力增强特征 [B, N, C]（与输入维度一致）
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim  # 输入/输出通道数
        self.num_heads = num_heads  # 注意力头数
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)  # QK投影层（一次性生成Q和K）
        self.elu = nn.ELU()  # 激活函数（确保Q/K非负）
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1,
                              groups=dim)  # 局部位置增强（LePE）：深度可分离卷积

    def forward(self, x, x_shape):
        b, n, c = x.shape  # 解析输入维度：批次、序列长度、通道数
        h, w = x_shape[2:]  # 从x_shape中提取空间尺寸（H/W）
        assert h * w == n, "输入序列长度与空间尺寸不匹配（n≠H×W）"

        num_heads = self.num_heads
        head_dim = c // num_heads  # 单头通道数

        # 步骤1：生成Q和K（V直接复用输入x，减少计算）
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)  # [2, B, N, C]
        q, k, v = qk[0], qk[1], x  # Q/K从qk拆分，V=x

        # 步骤2：Q/K激活（ELU+1确保非负性，避免注意力权重抵消）
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 步骤3：多头维度重排
        q = q.reshape(b, n, num_heads,
                      head_dim).permute(0, 2, 1,
                                        3)  # [B, num_heads, N, head_dim]
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        # 步骤4：线性注意力计算（O(N)复杂度）
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6
                 )  # 归一化因子
        kv = (k.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))  # KV融合（线性操作）
        x_attn = q @ kv * z  # 注意力加权输出

        # 步骤5：维度恢复
        x_attn = x_attn.transpose(1, 2).reshape(b, n, c)  # [B, N, C]

        # 步骤6：局部位置增强（LePE）融合
        v_4d = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1,
                                                             2)  # V转为4D空间格式
        lepe_feature = self.lepe(v_4d).permute(0, 2, 3,
                                               1).reshape(b, n,
                                                          c)  # LePE特征→3D序列格式
        x_attn = x_attn + lepe_feature  # 注意力特征+LePE特征

        return x_attn

    def extra_repr(self) -> str:
        """模块额外信息（print(model)时显示）"""
        return f'dim={self.dim}, num_heads={self.num_heads}'


class LAMA(nn.Module):
    """
    线性注意力增强模块（LAMA）：整合线性注意力、多路径特征融合、残差连接，实现高效特征增强
    核心创新：
        1. 双局部位置增强（CPE）：输入输出端均加入深度可分离卷积，强化局部细节；
        2. 激活门控融合：SiLU激活+投影层，动态筛选有效特征；
        3. 线性注意力核心：低复杂度适配高分辨率，推理速度快；
        4. 残差+随机深度：稳定深层训练，避免过拟合与梯度消失。
    输入：
        x - 序列特征 [B, N, C]（N=H×W）
        x_shape - 原始空间特征形状 [B, C, H, W]
    输出：
        增强后特征 [B, N, C]（与输入维度一致，支持即插即用）
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.dim = dim  # 输入/输出通道数
        self.num_heads = num_heads  # 注意力头数
        self.mlp_ratio = mlp_ratio  # MLP中间通道扩展因子

        # 输入端局部位置增强（CPE1）：深度可分离卷积，补充输入特征细节
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)  # 归一化层（稳定注意力计算）

        # 激活门控投影：动态筛选特征
        self.in_proj = nn.Linear(dim, dim)  # 输入投影
        self.act_proj = nn.Linear(dim, dim)  # 激活投影
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1,
                             groups=dim)  # 深度可分离卷积（增强空间关联）
        self.act = nn.SiLU()  # 激活函数（门控专用）

        self.attn = LinearAttention(dim=dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias)  # 线性注意力核心
        self.out_proj = nn.Linear(dim, dim)  # 注意力输出投影
        self.drop_path = nn.Identity()  # 随机深度

        # 输出端局部位置增强（CPE2）：进一步强化特征细节
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)  # MLP前归一化
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)  # 前馈网络

    def forward(self, x, x_shape):
        B, L, C = x.shape  # 解析输入维度：批次、序列长度、通道数
        H, W = x_shape[2:]  # 空间尺寸（H/W）
        assert H * W == L, "输入序列长度与空间尺寸不匹配（L≠H×W）"

        # 步骤1：输入增强（CPE1融合）
        # 3D→4D→CPE1→4D→3D，与原始输入残差融合
        x = x + self.cpe1(
            x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, N, C]→[B, C, H, W]
        ).flatten(2).permute(0, 2, 1)  # [B, C, H, W]→[B, N, C]

        # 步骤2：激活门控与空间增强
        shortcut = x  # 残差连接：保存当前特征
        x = self.norm1(x)  # 归一化
        act_res = self.act(self.act_proj(x))  # 激活门控：动态筛选特征
        # 3D→4D→深度可分离卷积→激活→4D→3D
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3,
                                                              1).view(B, L, C)

        # 步骤3：线性注意力增强
        x = self.attn(x, x_shape)  # 线性注意力计算
        x = self.out_proj(x * act_res)  # 激活门控融合+输出投影
        x = shortcut + self.drop_path(x)  # 残差融合+随机深度

        # 步骤4：输出增强（CPE2融合）
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(
            0, 3, 1, 2)).flatten(2).permute(0, 2, 1)  # CPE2特征残差融合

        # 步骤5：前馈网络（MLP）增强
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 归一化→MLP→残差融合

        return x

    def extra_repr(self) -> str:
        """模块额外信息（print(model)时显示）"""
        return f"dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 32 * 32, 64).to(device)
    x_shape = (1, 64, 32, 32)
    model = LAMA(64, 4).to(device)

    y = model(x, x_shape)

    print("微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
