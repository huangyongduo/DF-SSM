"""
核心模块定义 - DF-SSM PyTorch实现
包含：MAMLP (Multi-Axis Gated MLP)、CGB (Cross-Gating Block)、SAM等核心模块

核心模块说明：
1. GridGmlpLayer - 全局gMLP层，在grid轴上进行空间混合
2. BlockGmlpLayer - 局部gMLP层，在block轴上进行空间混合  
3. MultiAxisGmlpBlock (MAB) - 多轴门控MLP块，split-head并行处理局部+全局
4. CrossGatingBlock (CGB) - 交叉门控块，替代cross-attention实现跨特征条件化
5. SAM - 监督注意力模块，用于多阶段训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import (
    LayerNorm2d, Conv2dSame, ConvTranspose2dSame,
    block_images, unblock_images, weight_init
)


# ==================== 空间门控单元 ====================

class GridGatingUnit(nn.Module):
    """
    Grid空间门控单元
    在grid维度（第-3维）上进行空间混合
    Grid 门控单元 (axis=-3)
    
    输入形状: (B, num_grids, grid_size, C/2)
    门控作用在 num_grids 维度上
    """
    def __init__(self, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.norm = None  # 延迟初始化
        self.spatial_proj = None  # 延迟初始化，依赖空间维度(不知道图片大小，等图片传送再归一化）
    
    def forward(self, x):
        # x: (B, num_grids, grid_size, C)（C为通道数）
        b, n_grids, grid_size, c = x.shape
        
        # 通道分为两半: u, v
        u, v = torch.split(x, c // 2, dim=-1)
        c_half = c // 2
        
        if self.norm is None:
            self.norm = nn.LayerNorm(c_half, eps=1e-6).to(x.device)#如果没正则化则正则化
        
        if self.spatial_proj is None:#如果空间投影层没有创建则创建一个全连接层
            # 空间投影，作用在grid维度（第1维，索引-3）
            self.spatial_proj = nn.Linear(n_grids, n_grids, bias=self.use_bias).to(x.device)
            #它的输入和输出维度都是 n_grids（网格数量）。注意：这里是在“网格”这个空间维度上做全连接，而不是在通道上做！
            nn.init.normal_(self.spatial_proj.weight, std=0.02)#把权重初始化为均值为0，标准差为0.02的正态分布
            if self.use_bias:
                nn.init.ones_(self.spatial_proj.bias)  # 偏置初始化为1，对应v+1，初始为1可以保证训练刚开始时，门控是“打开”的，不会阻断梯度。
        
        v = self.norm(v)
        
        # swapaxes: 将grid维度放到最后，做Linear
        #把 v 的维度从 (批次, 网格数, 网格大小, 通道) 变成 (批次, 通道, 网格大小, 网格数)。把“网格数”移到了最后。
        # (B, num_grids, grid_size, C/2) -> (B, C/2, grid_size, num_grids)
        v = rearrange(v, 'b g s c -> b c s g')
        v = self.spatial_proj(v)#因为 nn.Linear 默认只对最后一个维度做全连接。移过去后，spatial_proj 就能在“网格数”这个维度上混合信息了（即全局空间混合）。
#做完 Linear 后，再用 rearrange 把维度“洗”回原来的顺序。
        # swap back
        v = rearrange(v, 'b c s g -> b g s c')
        
        return u * (v + 1.0)


class BlockGatingUnit(nn.Module):
    """
    Block空间门控单元
    在block维度（第-2维）上进行空间混合
    Block 门控单元 (axis=-2)
    
    输入形状: (B, num_blocks, block_size, C/2)
    门控作用在 block_size 维度上
    """
    def __init__(self, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.norm = None  # 延迟初始化
        self.spatial_proj = None  # 延迟初始化
    
    def forward(self, x):
        # x: (B, num_blocks, block_size, C)
        b, n_blocks, block_size, c = x.shape
        
        # 通道分为两半: u, v
        u, v = torch.split(x, c // 2, dim=-1)
        c_half = c // 2
        
        if self.norm is None:
            self.norm = nn.LayerNorm(c_half, eps=1e-6).to(x.device)
        
        if self.spatial_proj is None:
            # 空间投影，作用在block维度（第2维，索引-2）
            self.spatial_proj = nn.Linear(block_size, block_size, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.spatial_proj.weight, std=0.02)
            if self.use_bias:
                nn.init.ones_(self.spatial_proj.bias)
        
        v = self.norm(v)
        
        # swapaxes: 将block维度放到最后，做Linear
        # (B, num_blocks, block_size, C/2) -> (B, num_blocks, C/2, block_size)
        v = rearrange(v, 'b n s c -> b n c s')
        v = self.spatial_proj(v)
        # swap back
        v = rearrange(v, 'b n c s -> b n s c')
        
        return u * (v + 1.0)


# ==================== gMLP层 ====================

class GridGmlpLayer(nn.Module):
    """
    Grid gMLP层 - 全局空间混合
    将特征图按grid划分，在grid轴上进行gMLP操作
    
    Grid GMLP 层
    """
    def __init__(self, grid_size, use_bias=True, factor=2, dropout_rate=0.0):
        super().__init__()#grid_size：网格的尺寸，比如把图分成 8x8 的网格。
#factor：通道扩展倍数，MLP 通常先扩展通道再压缩。
        self.grid_size = grid_size  # (gh, gw)
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        
        self.norm = None  # 延迟初始化
        self.in_project = None
        self.gating_unit = GridGatingUnit(use_bias=use_bias)
        self.out_project = None
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: (B, C, H, W) NCHW格式
        b, c, h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw  # 每个grid的大小,输入 x 是标准的 PyTorch 图像格式：(批次, 通道, 高, 宽)。
#计算每个网格的物理像素大小 fh, fw。比如图高 256，分 8 个网格，每个网格高就是 32。
        
        # 分块: NCHW -> (B, GH*GW, FH*FW, C)
        x_blocks = block_images(x, patch_size=(fh, fw))
        _, n_blocks, block_size, _ = x_blocks.shape
        #block_images：把图像切成小块，并展平。形状变成 (批次, 网格总数, 网格内像素总数, 通道)。
        #上采样
        # 延迟初始化,根据输入通道数 c，初始化 LayerNorm、输入投影层（通道数扩大 factor 倍）、输出投影层（通道数压缩回 c）
        #eps防止标准化归零，bias为层归一化的可学习参数
        if self.norm is None:
            self.norm = nn.LayerNorm(c, eps=1e-6).to(x.device)
        if self.in_project is None:
            self.in_project = nn.Linear(c, c * self.factor, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.in_project.weight, std=0.02)
        if self.out_project is None:
            self.out_project = nn.Linear(c * self.factor // 2, c, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.out_project.weight, std=0.02)
        
        # gMLP前向
        y = self.norm(x_blocks)
        y = self.in_project(y)
        y = F.gelu(y)
        y = self.gating_unit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        
        # 残差连接
        x_blocks = x_blocks + y
        
        # 还原: (B, GH*GW, FH*FW, C) -> NCHW
        x = unblock_images(x_blocks, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGmlpLayer(nn.Module):
    """
    Block gMLP层 - 局部空间混合
    将特征图按block划分，在block轴上进行gMLP操作
    
    Block GMLP 层
    """
    def __init__(self, block_size, use_bias=True, factor=2, dropout_rate=0.0):
        super().__init__()
        self.block_size = block_size  # (fh, fw)
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        
        self.norm = None  # 延迟初始化
        self.in_project = None
        self.gating_unit = BlockGatingUnit(use_bias=use_bias)
        self.out_project = None
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: (B, C, H, W) NCHW格式
        b, c, h, w = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw  # 有多少个block
        
        # 分块: NCHW -> (B, GH*GW, FH*FW, C)
        x_blocks = block_images(x, patch_size=(fh, fw))
        _, n_blocks, block_size, _ = x_blocks.shape
        
        # 延迟初始化
        if self.norm is None:
            self.norm = nn.LayerNorm(c, eps=1e-6).to(x.device)
        if self.in_project is None:
            self.in_project = nn.Linear(c, c * self.factor, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.in_project.weight, std=0.02)
        if self.out_project is None:
            self.out_project = nn.Linear(c * self.factor // 2, c, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.out_project.weight, std=0.02)
        
        # gMLP前向
        y = self.norm(x_blocks)
        y = self.in_project(y)
        y = F.gelu(y)
        y = self.gating_unit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        
        # 残差连接
        x_blocks = x_blocks + y
        
        # 还原: (B, GH*GW, FH*FW, C) -> NCHW
        x = unblock_images(x_blocks, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


# ==================== 多轴门控MLP块 (MAMLP核心) ====================
#整个UV路线走法
class MultiAxisGmlpBlock(nn.Module):
    """
    多轴门控MLP块 (Multi-Axis Gated MLP Block, MAB)
    Split-head设计：通道分为两半，分别走Grid(全局)和Block(局部)分支
    
    这是DF-SSM的核心模块之一，实现局部+全局的并行空间混合 
    多轴 GMLP 残差块
    """
    def __init__(
        self, 
        block_size, 
        grid_size, 
        block_gmlp_factor=2, 
        grid_gmlp_factor=2,
        input_proj_factor=2, 
        use_bias=True, 
        dropout_rate=0.0
    ):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        
        self.norm_in = None  # 延迟初始化
        self.in_project = None
        self.grid_gmlp = GridGmlpLayer(
            grid_size=grid_size,
            factor=grid_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )
        self.block_gmlp = BlockGmlpLayer(
            block_size=block_size,
            factor=block_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )
        self.out_project = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, C, H, W)
        shortcut = x
        b, c, h, w = x.shape

        # 延迟初始化
        if self.norm_in is None:
            self.norm_in = LayerNorm2d(c).to(x.device)
        if self.in_project is None:
            self.in_project = nn.Conv2d(
                c, c * self.input_proj_factor, kernel_size=1, bias=self.use_bias
            ).to(x.device)
            nn.init.normal_(self.in_project.weight, std=0.02)
        # ========== 修复out_project通道 ==========
        if self.out_project is None:
            total_cat_ch = c * self.input_proj_factor  # 拼接后总通道 128
            self.out_project = nn.Conv2d(
                total_cat_ch, c, kernel_size=1, bias=self.use_bias
            ).to(x.device)
            nn.init.normal_(self.out_project.weight, std=0.02)

        # 输入归一化 + 投影
        x = self.norm_in(x)
        x = self.in_project(x)
        x = F.gelu(x)

        # Split-head: 通道分为两半
        c_proj = c * self.input_proj_factor
        u, v = torch.split(x, c_proj // 2, dim=1)

        # Grid分支（全局）
        u = self.grid_gmlp(u)
        # Block分支（局部）
        v = self.block_gmlp(v)
        # Concat两个分支 → 通道 = c_proj = 128
        x = torch.cat([u, v], dim=1)

        # 输出投影，输入128通道压缩回64
        x = self.out_project(x)
        x = self.dropout(x)

        # 残差连接
        x = x + shortcut
        return x


# ==================== 交叉门控块 (CGB核心) ====================

class GetSpatialGatingWeights(nn.Module):
    """
    计算交叉门控的空间权重
    用于CrossGatingBlock中，从输入特征计算门控权重
    
    空间门控权重计算
    """
    def __init__(
        self, 
        features, 
        block_size, 
        grid_size, 
        input_proj_factor=2, 
        dropout_rate=0.0, 
        use_bias=True
    ):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        self.norm_in = None
        self.in_project = None
        self.grid_proj = None  # grid分支的空间投影
        self.block_proj = None  # block分支的空间投影
        self.out_project = None
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        
        # 延迟初始化
        if self.norm_in is None:
            self.norm_in = LayerNorm2d(c).to(x.device)
        if self.in_project is None:
            self.in_project = nn.Conv2d(
                c, c * self.input_proj_factor,
                kernel_size=1, bias=self.use_bias
            ).to(x.device)
            nn.init.normal_(self.in_project.weight, std=0.02)
        if self.out_project is None:
            self.out_project = nn.Conv2d(
                c * self.input_proj_factor, c, kernel_size=1, bias=self.use_bias
            ).to(x.device)
            nn.init.normal_(self.out_project.weight, std=0.02)
        
        # 输入投影
        x = self.norm_in(x)
        x = self.in_project(x)
        x = F.gelu(x)
        
        # Split-head
        c_proj = c * self.input_proj_factor
        u, v = torch.split(x, c_proj // 2, dim=1)
        c_half = c_proj // 2
        
        # ===== Grid分支权重 =====
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw  # 每个网格格子尺寸
        u_blocks = block_images(u, patch_size=(fh, fw))
        dim_u = u_blocks.shape[1]  # GH*GW 网格总数
        # block_images拆分后Shape：[B, GH*GW, FH*FW, C]
        # 含义：GH×GW个网格，每个网格内FH×FW像素
        #延迟初始化grid_proj
        if self.grid_proj is None:
            self.grid_proj = nn.Linear(dim_u, dim_u, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.grid_proj.weight, std=0.02)
            if self.use_bias:
                nn.init.ones_(self.grid_proj.bias)
        # Linear作用在网格维度（交换维度实现单轴gMLP）
        u_swapped = rearrange(u_blocks, "b g s c -> b c s g")
        # Shape：[B, C, FH*FW, GH*GW]
        u_swapped = self.grid_proj(u_swapped)  # Linear(g,g) 网格间全局交互
        u_blocks = rearrange(u_swapped, "b c s g -> b g s c")
        # 还原分块格式 [B, GH*GW, FH*FW, C]

        u = unblock_images(u_blocks, grid_size=(gh, gw), patch_size=(fh, fw))
        # 还原原图尺寸 [B, C, H, W]
        #在网格数量维度做 MLP，让图像远距离网格像素产生交互，获得全局感受野。
        # ===== Block分支权重 =====
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw  # 局部窗口总数
        v_blocks = block_images(v, patch_size=(fh, fw))
        # Shape：[B, GH*GW, FH*FW, C]
        dim_v = v_blocks.shape[2]  # 单窗口像素数量
        # 【新增】延迟初始化block_proj
        if self.block_proj is None:
            self.block_proj = nn.Linear(dim_v, dim_v, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.block_proj.weight, std=0.02)
            if self.use_bias:
                nn.init.ones_(self.block_proj.bias)

        v_swapped = rearrange(v_blocks, "b n s c -> b n c s")
        # Shape：[B, GH*GW, C, FH*FW]
        v_swapped = self.block_proj(v_swapped)  # Linear(s,s) 窗口内局部交互
        v_blocks = rearrange(v_swapped, "b n c s -> b n s c")

        v = unblock_images(v_blocks, grid_size=(gh, gw), patch_size=(fh, fw))
        # 还原原图 [B, C, H, W]
        #在单窗口像素维度做 MLP，捕捉窗口内局部边缘、纹理
        # Concat两个分支的权重
        x = torch.cat([u, v], dim=1)
        
        # 输出投影
        x = self.out_project(x)
        x = self.dropout(x)
        
        return x


class CrossGatingBlock(nn.Module):
    """
    交叉门控块 (Cross-Gating Block, CGB)
    替代Cross-Attention，实现跨特征条件化
    
    两个输入特征x和y，互相计算门控权重：
    - X被Y的门控权重调制
    - Y被X的门控权重调制
    
    交叉门控块
    """
    def __init__(
        self, 
        features, 
        block_size, 
        grid_size, 
        dropout_rate=0.0,
        input_proj_factor=2, 
        upsample_y=True, 
        use_bias=True
    ):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        
        # Y上采样
        # Y上采样
        if upsample_y:
            # ✅ 新增：通道对齐投影层
            # global_feature 从 Bottleneck 出来是 512 通道，
            # 经过每一级 CGB 后会输出 features 通道，并传给下一级。
            # 所以下一级 CGB 收到的 y 永远是 当前 features 的 2 倍。
            self.y_proj = Conv2dSame(
                features * 2, features, kernel_size=1, bias=use_bias
            )

            self.upsample = ConvTranspose2dSame(
                features, features, kernel_size=2, stride=2, bias=use_bias
            )

        # 输入投影
        self.conv_x = Conv2dSame(features, features, kernel_size=1, bias=use_bias)
        self.conv_y = Conv2dSame(features, features, kernel_size=1, bias=use_bias)
        
        # LayerNorm
        self.norm_x = None  # 延迟初始化
        self.norm_y = None
        
        # 输入投影（Dense）
        self.in_proj_x = None
        self.in_proj_y = None
        
        # 门控权重计算
        self.gating_x = GetSpatialGatingWeights(
            features=features,
            block_size=block_size,
            grid_size=grid_size,
            input_proj_factor=input_proj_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias
        )
        self.gating_y = GetSpatialGatingWeights(
            features=features,
            block_size=block_size,
            grid_size=grid_size,
            input_proj_factor=input_proj_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias
        )
        
        # 输出投影
        self.out_proj_x = None
        self.out_proj_y = None
        self.dropout_x = nn.Dropout(dropout_rate)
        self.dropout_y = nn.Dropout(dropout_rate)
    
    def forward(self, x, y):
        """
        Args:
            x: 主特征 (B, C, H, W)
            y: 门控特征 (B, C, H/2, W/2) 或 (B, C, H, W)
            
        Returns:
            x_out, y_out: 交叉门控后的两个特征
        """
        b, c, h, w = x.shape
        
        # Y上采样（如果需要）
        # Y上采样（如果需要）
        if self.upsample_y:
            y = self.y_proj(y)  # ✅ 先对齐通道 (例: 512 → 256)
            y = self.upsample(y)  # 再 2 倍上采样
        # ConvTranspose2dSame 2倍上采样
        # y原[B,C,H/2,W/2] → 变成 [B,C,H,W]，和x尺寸完全匹配
        
        # 输入投影，统一通道数
        x = self.conv_x(x)
        y = self.conv_y(y)
        #残差连接
        shortcut_x = x
        shortcut_y = y
        
        # 延迟初始化
        if self.norm_x is None:
            self.norm_x = LayerNorm2d(c).to(x.device)
            self.norm_y = LayerNorm2d(c).to(x.device)
        if self.in_proj_x is None:
            self.in_proj_x = nn.Conv2d(c, c, kernel_size=1, bias=self.use_bias).to(x.device)
            self.in_proj_y = nn.Conv2d(c, c, kernel_size=1, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.in_proj_x.weight, std=0.02)
            nn.init.normal_(self.in_proj_y.weight, std=0.02)
        if self.out_proj_x is None:
            self.out_proj_x = nn.Conv2d(c, c, kernel_size=1, bias=self.use_bias).to(x.device)
            self.out_proj_y = nn.Conv2d(c, c, kernel_size=1, bias=self.use_bias).to(x.device)
            nn.init.normal_(self.out_proj_x.weight, std=0.02)
            nn.init.normal_(self.out_proj_y.weight, std=0.02)
        
        # 从X计算门控权重gx
        x_norm = self.norm_x(x)
        x_proj = self.in_proj_x(x_norm)
        x_proj = F.gelu(x_proj)
        gx = self.gating_x(x_proj)
        
        # 从Y计算门控权重gy
        y_norm = self.norm_y(y)
        y_proj = self.in_proj_y(y_norm)
        y_proj = F.gelu(y_proj)
        gy = self.gating_y(y_proj)
        
        # 交叉门控
        # Y = Y * GX  (Y被X的门控权重调制)
        y_gated = y_proj * gx
        y_out = self.out_proj_y(y_gated)
        y_out = self.dropout_y(y_out)
        y_out = y_out + shortcut_y
        
        # X = X * GY  (X被Y的门控权重调制)
        x_gated = x_proj * gy
        x_out = self.out_proj_x(x_gated)
        x_out = self.dropout_x(x_out)
        x_out = x_out + y_out + shortcut_x  # 聚合所有信号
        
        return x_out, y_out


# ==================== 监督注意力模块 (SAM) ====================

class SAM(nn.Module):
    """
    监督注意力模块 (Supervised Attention Module)
    用于多阶段训练，来自MPRNet (CVPR2021)
    
    功能：
    1. 输出当前阶段的恢复图像
    2. 生成注意力特征图，传递给下一阶段
    
    监督注意力模块 (SAM)
    """
    def __init__(self, features, output_channels=3, use_bias=True):
        super().__init__()
        self.features = features
        self.output_channels = output_channels
        self.use_bias = use_bias
        
        self.conv_features = Conv2dSame(features, features, kernel_size=3, bias=use_bias)
        self.conv_output = Conv2dSame(features, output_channels, kernel_size=3, bias=use_bias)
        self.conv_attention = Conv2dSame(output_channels, features, kernel_size=3, bias=use_bias)
    
    def forward(self, x, x_image):
        """
        Args:
            x: Decoder输出特征 (B, C, H, W)
            x_image: 输入图像 (B, 3, H, W)
            
        Returns:
            sam_features: 传递给下一阶段的注意力特征
            output_image: 当前阶段的输出恢复图像
        """
        # 特征提取
        x1 = self.conv_features(x)
        
        # 输出恢复图像
        if self.output_channels == 3:
            output_image = self.conv_output(x) + x_image
        else:
            output_image = self.conv_output(x)
        # self.conv_output(x)：C 通道特征经过 3×3 卷积，映射为 3 通道图像残差
        # Shape：[B, C, H, W] → [B, 3, H, W]
        # 图像修复残差 + 原始退化输入x_image：残差学习范式
        # 网络只需要学习 “退化到干净图像的差值”，收敛更快；
        # 输出output_image：本阶段重建结果，送入多尺度 Charbonnier Loss 计算监督损失。
        # 最终 Shape：output_image = [B, 3, H, W]
        # 注意力图
        attention = torch.sigmoid(self.conv_attention(output_image))
        # self.conv_attention(output_image)：3 通道修复图卷积升维回 C 通道
        # Shape[B, 3, H, W] → [B, C, H, W]
        # torch.sigmoid()：逐像素压缩至(0,1)区间，得到注意力权重掩码；
        # 语义：掩码数值越高，代表该像素区域修复效果越好、有效信息越多；
        # Shape 不变：attention = [B, C, H, W]
        
        # 注意力调制
        x1 = x1 * attention
        
        # 残差连接
        sam_features = x1 + x
        
        return sam_features, output_image#sam_features：送入下一 Stage 的 CGB 交叉门控块，作为跨阶段辅助特征；
        #output_image：本尺度重建图，参与多尺度多阶段 Loss 加权求和。


# ==================== Bottleneck块 ====================

class BottleneckBlock(nn.Module):
    """
    Bottleneck块
    包含：多轴gMLP块 + RDCAB通道注意力
    
    Bottleneck 块
    """
    def __init__(
        self, 
        features, 
        block_size, 
        grid_size, 
        num_groups=1,
        block_gmlp_factor=2, 
        grid_gmlp_factor=2,
        input_proj_factor=2, 
        channels_reduction=4,
        dropout_rate=0.0, 
        use_bias=True
    ):
        super().__init__()
        self.features = features
        self.num_groups = num_groups
        
        # 输入投影
        self.input_proj = Conv2dSame(features, features, kernel_size=1, bias=use_bias)
        
        # 多组MAB + RDCAB
        self.mab_blocks = nn.ModuleList()
        self.rcab_blocks = nn.ModuleList()
        
        for i in range(num_groups):
            self.mab_blocks.append(
                MultiAxisGmlpBlock(
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    use_bias=use_bias,
                    dropout_rate=dropout_rate
                )
            )
            self.rcab_blocks.append(
                RDCAB(
                    features=features,
                    reduction=channels_reduction,
                    use_bias=use_bias,
                    dropout_rate=dropout_rate
                )
            )

    def forward(self, x):
        # ========== 步骤1：输入1×1投影 ==========
        x = self.input_proj(x)
        # 操作：1×1 TF-SAME卷积，kernel=1、stride=1，通道/高宽完全不变
        # Shape：[B, C, H, W] → [B, C, H, W]

        # 保存长残差shortcut，最后融合
        shortcut_long = x

        # ========== 步骤2：循环每组 MAB + RDCAB ==========
        for i in range(self.num_groups):
            # 第1步：多轴门控MLP 空间混合（局部+全局并行分支）
            x = self.mab_blocks[i](x)
            # MAB内部全程不改变C/H/W，输出仍 [B, C, H, W]

            # 第2步：RDCAB 通道混合+SE注意力
            x = self.rcab_blocks[i](x)
            # RDCAB输入输出尺寸一致，输出 [B, C, H, W]
        # 循环完成：多层空间+通道特征提纯

        # ========== 步骤3：长残差连接融合 ==========
        x = x + shortcut_long
        # 逐元素相加：原始输入投影特征 + 多层MAB/RDCAB处理后的特征
        # Shape不变：[B, C, H, W]
        return x


# ==================== APDM模块 (Amplitude-Phase Decoupling Module) ====================

class APDM(nn.Module):
    """
    APDM (Amplitude-Phase Decoupling Module)
    频域引导模块 - 输出空间注意力权重图，而非空间特征恢复
    
    核心设计思想：
    1. 将输入特征经FFT转换到频域，捕获全局频率信息
    2. 对幅度谱进行频域处理，生成通道无关的空间注意力图
    3. 输出注意力权重W_freq，指导MAB空间特征的自适应增强/抑制
    4. 使用Dropout2d(p=0.2)抑制过拟合
    
    输出：W_freq (B, 1, H, W) - 空间注意力图，值域[0,1]
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        hidden_channels = max(channels // reduction, 8)
        
        # ========== 频域处理分支 ==========
        # 幅度谱处理：提取频域全局信息
        self.amp_conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.amp_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        
        # ========== 正则化层 ==========
        # Dropout2d在频域处理后添加，有效抑制过拟合
        # p=0.2表示随机丢弃20%的通道特征
        self.dropout = nn.Dropout2d(p=0.2)
        
        # ========== 注意力权重生成 ==========
        # 1x1卷积将频域特征投影到单通道注意力图
        # 使用Sigmoid将值域限制在[0,1]，作为空间门控系数
        self.attn_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化注意力卷积为接近1的输出（恒等初始化）
        nn.init.zeros_(self.attn_conv.weight)
        nn.init.constant_(self.attn_conv.bias, 5.0)  # sigmoid(5) ≈ 0.993，初始接近1
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            W_freq: 空间注意力权重图 (B, 1, H, W)，值域[0,1]
            
        注意：
        - 移除了原有的IFFT和空间特征恢复功能
        - 仅输出频域引导的空间注意力权重
        - FFT操作强制使用FP32精度以保证数值稳定性
        - 不使用切断梯度的回退逻辑（NaN/Inf由训练循环处理）
        """
        b, c, h, w = x.shape
        input_dtype = x.dtype
        
        # 使用FP32进行FFT运算，避免FP16精度不足
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            
            # ========== 步骤1：FFT转换到频域 ==========
            x_fft = torch.fft.fft2(x_fp32, norm='ortho')
            del x_fp32
            
            # ========== 步骤2：幅度谱处理 ==========
            amplitude = torch.abs(x_fft)
            del x_fft
            
            # 关键修复：使用log压缩幅度谱，防止数值过大
            # log(1 + x)可以压缩动态范围，同时保留梯度信息
            amplitude = torch.log1p(amplitude)
            
            # 对幅度谱做clamp保护（仅限制极大值，不影响梯度）
            amplitude = torch.clamp(amplitude, min=1e-8, max=100.0)
            
            # 频域特征提取：通道压缩 -> ReLU -> 通道扩展
            freq_feat = self.amp_conv1(amplitude)
            freq_feat = self.relu(freq_feat)
            freq_feat = self.amp_conv2(freq_feat)
            del amplitude
        
        # 恢复到原始数据类型
        freq_feat = freq_feat.to(input_dtype)
        
        # ========== 步骤3：Dropout正则化 ==========
        freq_feat = self.dropout(freq_feat)
        
        # ========== 步骤4：生成空间注意力权重图 ==========
        W_freq = self.attn_conv(freq_feat)
        
        # Sigmoid激活：将注意力权重限制在[0,1]范围
        # sigmoid本身就有数值范围保护，不需要额外clamp
        W_freq = self.sigmoid(W_freq)
        
        return W_freq  # (B, 1, H, W)


# ==================== 带APDM的Bottleneck块 ====================

class BottleneckWithAPDM(nn.Module):
    """
    带APDM频域引导的Bottleneck块
    
    结构：频域引导空间注意力（Frequency-guided Spatial Attention）
    
    核心融合逻辑（级联结构）：
    1. MAB分支输出空间特征 F_spatial (B, C, H, W)
    2. APDM分支输出空间注意力图 W_freq (B, 1, H, W)，值域[0,1]
    3. 融合公式：Output = F_spatial * W_freq + F_spatial
       - 频域信息指导空间特征中不同区域的增强或抑制
       - W_freq≈1时保持原特征，W_freq≈0时抑制该区域
       - 残差连接保证梯度流畅，避免性能退化
    
    相比原并联融合的优势：
    - 消除特征冲突：APDM仅输出注意力权重，不直接修改特征
    - 频域引导更自然：空间特征为主，频域信息作为指导信号
    - 输出恒等初始化：训练初期W_freq≈1，模型从原Bottleneck开始优化
    """
    def __init__(
        self, 
        features, 
        block_size, 
        grid_size, 
        num_groups=1,
        block_gmlp_factor=2, 
        grid_gmlp_factor=2,
        input_proj_factor=2, 
        channels_reduction=4,
        dropout_rate=0.0, 
        use_bias=True
    ):
        super().__init__()
        self.features = features
        
        # ========== MAB分支：原有BottleneckBlock ==========
        # 输出空间域特征 F_spatial (B, C, H, W)
        self.mab_branch = BottleneckBlock(
            features=features,
            block_size=block_size,
            grid_size=grid_size,
            num_groups=num_groups,
            block_gmlp_factor=block_gmlp_factor,
            grid_gmlp_factor=grid_gmlp_factor,
            input_proj_factor=input_proj_factor,
            channels_reduction=channels_reduction,
            dropout_rate=dropout_rate,
            use_bias=use_bias
        )
        
        # ========== APDM分支：频域引导模块 ==========
        # 输出空间注意力权重图 W_freq (B, 1, H, W)
        self.apdm_branch = APDM(
            channels=features,
            reduction=channels_reduction
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            output: 频域引导融合后的特征 (B, C, H, W)
            
        维度变化说明：
        - MAB输出: (B, C, H, W)
        - APDM输出: (B, 1, H, W)
        - 融合后输出: (B, C, H, W) - 通道数保持不变
        
        注意：
        - 不使用切断梯度的回退逻辑
        - 通过clamp保护数值稳定性
        """
        # ========== 分支1：MAB空间特征提取 ==========
        F_spatial = self.mab_branch(x)
        
        # ========== 分支2：APDM频域注意力生成 ==========
        W_freq = self.apdm_branch(x)
        
        # ========== 频域引导融合 ==========
        # 使用broadcast机制，避免显式expand
        # Output = F_spatial * (W_freq + 1)
        output = F_spatial * (W_freq + 1.0)
        
        # 数值稳定性：对输出做clamp，不切断梯度
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # 释放中间张量（仅释放已使用的引用）
        del F_spatial, W_freq
        
        return output


# 从layers导入，放在最后避免循环导入
from .layers import RDCAB
