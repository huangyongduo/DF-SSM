"""
基础层定义 - DF-SSM PyTorch实现
包含：卷积层、归一化层、MLP、通道注意力等基础组件

注意：
- 内部张量按 NCHW 布局，MLP 模块内部做维度转换
- LayerNorm在通道维度计算时需要permute或使用GroupNorm(1, C)
- 使用einops进行灵活的维度重排
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange


# ==================== 工具函数 ====================

def weight_init(m):
    """
    权重初始化
    Conv和Linear使用正态分布初始化，std=0.02
    """
    # 第1行：判断当前层是否是2D卷积 或 转置卷积（上采样卷积）
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.02)
        # 第2行：权重正态分布初始化，均值0，标准差0.02（TF默认初始化参数）
        if m.bias is not None:# 存在偏置项则全部置零
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):    # 第5行：判断当前层是全连接Linear层（MLP、门控单元专用）
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def same_padding(kernel_size, stride=1, dilation=1):
    """
    计算 'SAME' padding
    处理偶数kernel_size时的不对称padding
    """
    if isinstance(kernel_size, int): # 第1行：判断输入卷积核是否为数字，转为(高度,宽度)二元组统一格式
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    pad_h = max((kernel_size[0] - 1) * dilation[0] + 1 - stride[0], 0) # 高度总填充像素
    pad_w = max((kernel_size[1] - 1) * dilation[1] + 1 - stride[1], 0)
    
    # 不对称padding：右边和下边多pad 1（对齐TF的SAME）
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    return (pad_left, pad_right, pad_top, pad_bottom)# 返回填充四元组，顺序适配F.pad


class Conv2dSame(nn.Module):
    """
    实现 SAME padding 卷积
    手动计算padding后使用F.pad + Conv2d(padding=0)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dilation=1, groups=1, bias=True):
        super().__init__()
        #保留为二元组
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=0, dilation=dilation, 
            groups=groups, bias=bias
        )
    
    def forward(self, x):
        padding = same_padding(self.kernel_size, self.stride, self.dilation)
        # ★ AMP 兼容: reflect pad + conv 全程在 FP32 下运行
        orig_dtype = x.dtype
        x = x.float()
        x = F.pad(x, padding, mode='reflect')
        x = self.conv(x)
        return x.to(orig_dtype)


class ConvTranspose2dSame(nn.Module):
    """
    转置卷积，用于上采样
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, bias=bias
        )
        #和普通 nn.Conv2d 正向卷积做相反运算：正向卷积通常缩小特征图，转置卷积用来放大特征图尺寸、恢复分辨率
#计算公式（默认无 padding、dilation=1）：
#\(OutputSize = (InputSize - 1) \times stride + kernel\_size\)
#默认参数：输入尺寸 N×C_in×H×W → 输出 N×C_out×(H-1)*2+2 × (W-1)*2+2 = 刚好 2 倍尺寸
#例：输入 64×64 → 输出 128×128，同时完成通道数变更
    
    def forward(self, x):
        return self.conv_transpose(x)


# ==================== 归一化层 ====================

class LayerNorm2d(nn.Module):
    """
    2D特征图的LayerNorm，在通道维度上归一化
    输入: NCHW格式
    实现方式: permute到NHWC -> LayerNorm -> permute回NCHW
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps) # 创建原生LN，归一通道维度
    
    def forward(self, x):
        # NCHW -> NHWC
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        # NHWC -> NCHW
        x = rearrange(x, 'b h w c -> b c h w')
        return x


# ==================== MLP层 ====================

class MlpBlock(nn.Module):
    """
    单隐藏层MLP块，在最后一个维度上应用
    输入: NCHW格式，内部转换为(B, H*W, C)进行计算
    """

    def __init__(
        self, mlp_dim, dropout_rate=0.0, use_bias=True):  # 构造：隐藏层维度、dropout、是否偏置
        super().__init__()  # 父类初始化
        self.mlp_dim = mlp_dim  # 保存MLP隐藏通道维度
        self.dropout_rate = dropout_rate  # 保存dropout概率
        self.use_bias = use_bias  # 保存Linear是否启用偏置
        self.net = None  # 延迟初始化，首次forward才创建网络

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape#(B, H*W, C)
        
        if self.net is None:
            self.net = nn.Sequential(
                nn.Linear(c, self.mlp_dim, bias=self.use_bias),#B,N,C->B,N,mlp_dim
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.mlp_dim, c, bias=self.use_bias),
                nn.Dropout(self.dropout_rate),
            ).to(x.device)
            # 应用权重初始化
            self.net.apply(weight_init) #对MLP执行权重初始化
        
        # NCHW -> (B, H*W, C)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.net(x_flat)
        # (B, H*W, C) -> NCHW
        x = rearrange(x_flat, 'b (h w) c -> b c h w', h=h, w=w)
        return x


# ==================== 通道注意力 ====================

class CALayer(nn.Module):
    """
    Squeeze-and-Excitation通道注意力块
    参考: https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self, features, reduction=4, use_bias=True):  # 构造：通道数、压缩倍率、偏置
        super().__init__()  # 父类初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，压缩单通道为1×1
        self.fc = nn.Sequential(  # 两层1×1卷积实现挤压激励
            nn.Conv2d(
                features, features // reduction, kernel_size=1, bias=use_bias
            ),  # 压缩通道
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv2d(
                features // reduction, features, kernel_size=1, bias=use_bias
            ),  # 恢复通道
            nn.Sigmoid(),  # 输出0~1通道权重
        )

    def forward(self, x):  # 前向传播
        y = self.avg_pool(x)  # 全局挤压，得到各通道全局描述
        y = self.fc(y)  # 学习通道依赖，生成权重
        return x * y  # 原图 × 通道权重，自适应加权

class RCAB(nn.Module):
    """
    残差通道注意力块 (Residual Channel Attention Block)
    包含: LayerNorm -> Conv -> LeakyReLU -> Conv -> CALayer -> 残差连接
    """
    def __init__(self, features, reduction=4, lrelu_slope=0.2, use_bias=True):  # 构造超参
        super().__init__()  # 父类初始化
        self.norm = LayerNorm2d(features)  # 图像层归一化
        self.conv1 = Conv2dSame(features, features, kernel_size=3, bias=use_bias)  # 第一层3×3 SAME卷积
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_slope, inplace=True)  # LeakyReLU激活
        self.conv2 = Conv2dSame(features, features, kernel_size=3, bias=use_bias)  # 第二层3×3 SAME卷积
        self.ca = CALayer(features, reduction=reduction, use_bias=use_bias)  # SE注意力
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.ca(x)
        return x + shortcut


class RDCAB(nn.Module):
    """
    残差稠密通道注意力块 (Residual Dense Channel Attention Block)
    用于Bottleneck中，使用MLP进行通道混合
    """
    def __init__(self, features, reduction=16, use_bias=True, dropout_rate=0.0):
        super().__init__()
        self.norm = LayerNorm2d(features)
        self.channel_mixing = MlpBlock(
            mlp_dim=features, 
            dropout_rate=dropout_rate, 
            use_bias=use_bias
        )
        self.channel_attention = CALayer(
            features=features, 
            reduction=reduction, 
            use_bias=use_bias
        )

    def forward(self, x):  # 前向
        shortcut = x  # 残差分支
        y = self.norm(x)  # 归一化
        y = self.channel_mixing(y)  # MLP通道混合
        y = self.channel_attention(y)  # 注意力加权
        return x + y  # 残差融合


# ==================== 图像块操作 ====================

def block_images(x, patch_size):
    """
    将图像分割为不重叠的patch
    输入: (B, C, H, W)
    输出: (B, num_patches, patch_size_h*patch_size_w, C)
    
    图像分块 (block) 变换
    """
    b, c, h, w = x.shape
    ph, pw = patch_size
    gh, gw = h // ph, w // pw
    
    # NCHW -> (B, GH*GW, PH*PW, C)
    x = rearrange(
        x, 'b c (gh ph) (gw pw) -> b (gh gw) (ph pw) c',
        gh=gh, gw=gw, ph=ph, pw=pw
    )
    return x


def unblock_images(x, grid_size, patch_size):
    """
    将patch还原为图像
    输入: (B, num_patches, patch_size_h*patch_size_w, C)
    输出: (B, C, H, W)
    
    图像还原 (unblock) 变换
    """
    gh, gw = grid_size
    ph, pw = patch_size
    
    # (B, GH*GW, PH*PW, C) -> NCHW
    x = rearrange(
        x, 'b (gh gw) (ph pw) c -> b c (gh ph) (gw pw)',
        gh=gh, gw=gw, ph=ph, pw=pw
    )
    return x


# ==================== 上采样 ====================

class UpSampleRatio(nn.Module):
    """
    按比例上采样特征图
    双线性插值 + 1x1卷积调整通道数
    """
    def __init__(self, features, ratio, use_bias=True):
        super().__init__()
        self.ratio = ratio  # 保存缩放倍率
        self.conv = Conv2dSame(
            features, features, kernel_size=1, bias=use_bias
        )  # 1×1平滑卷积

    def forward(self, x):  # 前向
        b, c, h, w = x.shape  # 读取原图高宽
        new_h, new_w = int(h * self.ratio), int(w * self.ratio)  # 计算缩放后尺寸
        x = F.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )  # 双线性插值缩放
        x = self.conv(x)  # 1×1卷积平滑特征
        return x  # 返回缩放特征