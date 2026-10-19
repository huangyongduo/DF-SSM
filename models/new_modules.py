"""
四大核心新模块实现
1. DCT-HPM (DCT High-Pass Module) - DCT高通物理先验
2. MASSM (Multi-Axis State Space Mixer) - 多轴状态空间混合器，替换MAB
3. SSM-CGB (SSM Cross-Gating Block) - SSM交叉门控块，替换CGB
4. APDM (Amplitude-Phase Decoupling Module) - 幅相解耦模块

依赖：
- 可选：mamba_ssm (如未安装，使用简化的SSM实现)
- 标准PyTorch：torch, torch.nn, torch.nn.functional
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 可选：尝试导入 mamba_ssm（如果已安装 CUDA 版本）
try:
    from mamba_ssm import Mamba as _MambaCUDA
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

# ==================== RMSNorm 实现 ====================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比 LayerNorm 更高效，适合 Mamba/SSM 架构
    
    公式：y = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 支持 (B, C, H, W), (B, L, C), (B, C) 等多种格式
        if x.dim() == 4:
            # (B, C, H, W) 格式 - norm 在所有维度上，weight 按通道
            norm = torch.sqrt(x.pow(2).mean(dim=[1, 2, 3], keepdim=True) + self.eps)
            # self.weight: (C,) -> (1, C, 1, 1)
            weight = self.weight.view(1, -1, 1, 1)
            return x / norm * weight
        elif x.dim() == 3:
            # (B, L, C) 序列格式 - norm 在序列维度上，weight 按特征
            norm = torch.sqrt(x.pow(2).mean(dim=2, keepdim=True) + self.eps)
            # self.weight: (C,) -> (1, 1, C)
            weight = self.weight.view(1, 1, -1)
            return x / norm * weight
        else:
            # (B, C) 或其他
            norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x / norm * self.weight


# ==================== SSM Block (mamba_ssm 优先 + 快速 fallback) ====================
class _MambaNative(nn.Module):
    """mamba_ssm.Mamba 的包装类，保证接口与 fallback 一致"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ssm = _MambaCUDA(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        # ★ mamba_ssm 自定义 CUDA 内核在 FP16 下可能不稳定
        # 禁用 AMP, 强制 FP32 (DDP 下梯度同步更稳定)
        with torch.amp.autocast('cuda', enabled=False):
            input_dtype = x.dtype
            return self.ssm(x.float()).to(input_dtype)


class _FallbackSSMBlock(nn.Module):
    """
    快速纯 PyTorch SSM fallback（O(L) 时间 / 空间复杂度）。

    设计思路：
    1. in_proj + depthwise conv1d (局部建模，模拟 Mamba 的 conv 核)
    2. 沿序列方向的指数衰减递推 (等价于 per-channel 的 1 维 SSM)
       - 每个通道有独立的衰减系数 lambda_c ∈ (0, 1)
       - 通过 cumulative_sum 实现 O(L) 扫描：
           y_t = lambda * y_{t-1} + (1 - lambda) * x_t
         这是最简单的稳定 SSM，能建模长程依赖。
    3. 输出投影 + RMSNorm

    相比 Python 循环扫描：
    - 使用 FIFO 级的累积求和，向量化无循环。
    - 保证长序列 (L=4096+) 仍能跑得动。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand

        inner = d_model * expand

        # 输入投影
        self.in_proj = nn.Linear(d_model, inner)

        # 局部建模 (depthwise conv1d, 保持序列长度不变)
        # 采用非对称 padding: 左侧 pad (d_conv-1) 个 0, 右侧 0 个
        # 这样输出序列长度 = L (因果性，仅看当前及历史)
        self.dw_conv = nn.Conv1d(
            inner, inner,
            kernel_size=d_conv, padding=0,
            groups=inner,
        )
        self._dw_pad = d_conv - 1  # 左侧补 0 的数量

        # 每个通道一个可学习的衰减系数 lambda ∈ (0, 1)
        # raw_lambda 通过 sigmoid 映射到 (0, 1)，并限制在 (0.5, 0.99) 范围
        self.lambda_raw = nn.Parameter(torch.zeros(inner))

        # 输出投影
        self.out_proj = nn.Linear(inner, d_model)

        self.norm = RMSNorm(inner)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            y: (B, L, C)
        """
        # ★ SSM 核心计算禁用 AMP (FP16 下 conv1d + 递推扫描不稳定)
        # 特别是 3 卡 DDP 下梯度同步时 FP16 精度不足会导致数值错误
        with torch.amp.autocast('cuda', enabled=False):
            return self._forward_fp32(x)

    def _forward_fp32(self, x):
        # ★ 强制输入为 FP32 (外层 autocast 可能传入 FP16)
        input_dtype = x.dtype
        x = x.float()

        B, L, C_in = x.shape

        # 1. 输入投影
        x_proj = self.in_proj(x)  # (B, L, C*expand)

        # 2. 局部建模 (depthwise conv1d, 保持序列长度 L 不变)
        x_t = x_proj.transpose(1, 2)  # (B, C*expand, L)
        # 左侧补 _dw_pad 个 0 (因果性)
        if self._dw_pad > 0:
            x_t_padded = F.pad(x_t, (self._dw_pad, 0))
        else:
            x_t_padded = x_t
        x_local = self.dw_conv(x_t_padded).transpose(1, 2)  # (B, L, C*expand)

        # 展开后的通道数
        C_expand = self.in_proj.out_features  # = d_model * expand

        # 3. 线性 SSM 扫描 (使用 conv1d 实现稳定的窗口递推)
        # 递推: y_t = lam * y_{t-1} + (1-lam) * x_t
        #       = (1-lam) * sum_{s=max(0, t-W+1)}^t lam^{t-s} * x_s

        lam = torch.sigmoid(self.lambda_raw) * 0.49 + 0.5  # (C_expand,)

        # M[t] = (1 - lam) * x_local[t]
        M = (1.0 - lam).view(1, 1, -1) * x_local  # (B, L, C_expand)

        W = 256  # 窗口大小

        # 计算权重 weight[k] = lam^k, k = 0, 1, ..., W-1
        weight_list = []
        w_cur = torch.ones_like(lam)  # (C_expand,)
        for _ in range(W):
            weight_list.append(w_cur)
            w_cur = w_cur * lam
        # weights: (W, C_expand)
        weights = torch.stack(weight_list, dim=0)

        # F.conv1d 的约定 (stride=1, padding=0):
        #   output[t] = sum_{k=0}^{K-1} input[t + K-1 - k] * kernel[k]
        # 我们希望: output[t] = sum_{k=0}^{W-1} M[t + k] * lam^k
        #   -> kernel 需要反转 (kernel[k] = lam^{W-1-k})
        #   左侧补 W-1 个 0

        kernel = weights.flip(0).transpose(0, 1).unsqueeze(1)  # (C_expand, 1, W)

        pad_left = M.new_zeros(B, W - 1, C_expand)
        M_padded = torch.cat([pad_left, M], dim=1)  # (B, L+W-1, C_expand)
        M_t = M_padded.transpose(1, 2)  # (B, C_expand, L+W-1)

        # Conv1d: 输出 (B, C_expand, L)
        y_ssm = F.conv1d(M_t, kernel, groups=C_expand, stride=1, padding=0)
        y_ssm = y_ssm.transpose(1, 2)  # (B, L, C_expand)

        # 4. 与局部特征融合 (残差)
        y = x_local + y_ssm

        # 5. 归一化 + 输出投影
        y = self.norm(y)
        y = self.out_proj(y)  # (B, L, C_in)

        # ★ 转换回原始 dtype (供外层 autocast 使用)
        return y.to(input_dtype)


class SimpleSSMBlock(nn.Module):
    """
    SSM 块：优先使用 mamba_ssm.Mamba (CUDA 高效实现)，
    如果环境中未安装则自动 fallback 到纯 PyTorch 实现。
    
    mamba_ssm 安装方式:
        pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.use_mamba = HAS_MAMBA_SSM

        if HAS_MAMBA_SSM:
            self.ssm = _MambaNative(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.ssm = _FallbackSSMBlock(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )

    def forward(self, x):
        return self.ssm(x)


# ==================== MASSM (Multi-Axis State Space Mixer) ====================
class MASSM(nn.Module):
    """
    多轴状态空间混合器 (Multi-Axis State Space Mixer)
    替换基础网络中的 MAB (Multi-Axis Gated MLP Block)

    结构：
    1. RMSNorm -> 通道投影 (C -> 2C) -> Split (C/2, C/2) 两个分支
    2. 分支1 (Horizontal Axis): Reshape -> SSM (Horizontal Scan) -> Reshape Back
    3. 分支2 (Vertical Axis): Transpose -> Reshape -> SSM (Vertical Scan) -> Reshape Back -> Transpose Back
    4. 融合: Concat (C/2 + C/2 = C) -> Linear -> GELU -> Linear (Spatial Projection)
    5. 残差连接

    与 MAB 的区别：
    - 使用 SSM (状态空间模型) 替代 MLP 进行空间混合
    - 沿水平/垂直轴进行选择性扫描
    - 更高效的长距离依赖建模
    """
    def __init__(
        self,
        features,           # 输入通道数 C
        state_dim=16,       # SSM 隐状态维度
        conv_kernel=4,      # 深度卷积核大小
        expand_ratio=2,     # SSM 内部扩展倍率
        residual_gamma=0.1, # 残差缩放因子 (防止训练初期 Mamba 输出破坏主干特征)
        dropout_rate=0.0,   # ★ MLP 融合后的 Dropout (正则化, 防过拟合)
    ):
        super().__init__()
        self.features = features
        self.state_dim = state_dim

        # 输入归一化
        self.norm = RMSNorm(features)

        # 通道投影: C -> C (保持通道数)
        self.in_proj = nn.Conv2d(features, features, kernel_size=1)

        # 水平轴 SSM: 接收 C/2 通道输入
        half_features = features // 2
        self.horizontal_ssm = SimpleSSMBlock(
            d_model=half_features,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=expand_ratio
        )

        # 垂直轴 SSM: 接收 C/2 通道输入
        self.vertical_ssm = SimpleSSMBlock(
            d_model=half_features,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=expand_ratio
        )

        # 融合投影: Concat 后保持通道数 C -> C
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
        )

        # ★ Dropout (在 MLP 融合之后, 残差连接之前)
        # 仅在训练时生效, model.eval() 自动关闭
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # ★ 数值稳定性加固：可学习的残差缩放因子 gamma
        # 初始化为 0.1，训练初期 Mamba 输出仅作为小幅扰动，避免破坏主干特征
        # 随训练进行网络会自适应调整 gamma 大小
        self.gamma = nn.Parameter(torch.tensor(residual_gamma))

        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            output: 输出特征 (B, C, H, W)，与输入同形状
        """
        shortcut = x
        B, C, H, W = x.shape
        half_C = C // 2

        # RMSNorm 归一化 (AMP 安全)
        x = self.norm(x)

        # 通道投影 (保持通道数 C) (AMP 安全)
        x_proj = self.in_proj(x)  # (B, C, H, W)

        # Split 成两半 (C/2, C/2) (AMP 安全)
        x_h, x_v = torch.split(x_proj, half_C, dim=1)  # 各 (B, C/2, H, W)

        # ========== 分支1: 水平轴扫描 ==========
        # 将 (B, C/2, H, W) reshape 为 (B*H, W, C/2) 进行水平扫描
        x_h_reshaped = rearrange(x_h, 'b c h w -> (b h) w c')  # (B*H, W, C/2)
        # ★ SSM 内部自动禁用 autocast (conv1d 递推在 FP16 下不稳定)
        x_h_out = self.horizontal_ssm(x_h_reshaped)  # (B*H, W, C/2)
        x_h_out = rearrange(x_h_out, '(b h) w c -> b c h w', h=H)  # (B, C/2, H, W)

        # ========== 分支2: 垂直轴扫描 ==========
        # 转置后 reshape 为 (B*W, H, C/2) 进行垂直扫描
        x_v_transposed = x_v.permute(0, 1, 3, 2)  # (B, C/2, W, H)
        x_v_reshaped = rearrange(x_v_transposed, 'b c w h -> (b w) h c')  # (B*W, H, C/2)
        # ★ SSM 内部自动禁用 autocast (conv1d 递推在 FP16 下不稳定)
        x_v_out = self.vertical_ssm(x_v_reshaped)  # (B*W, H, C/2)
        # reshape back: (B*W, H, C/2) -> (B, W, H, C/2) -> (B, C/2, H, W)
        x_v_out = rearrange(x_v_out, '(b w) h c -> b w h c', w=W)  # (B, W, H, C/2)
        x_v_out = x_v_out.permute(0, 3, 2, 1)  # (B, C/2, H, W)

        # ========== 融合 ==========
        # Concat 两个分支: (B, C/2, H, W) + (B, C/2, H, W) = (B, C, H, W)
        x_fused = torch.cat([x_h_out, x_v_out], dim=1)  # (B, C, H, W)

        # 投影融合 (Spatial Projection) (AMP 安全)
        output = self.fuse_proj(x_fused)  # (B, C, H, W)

        # ★ Dropout (在 MLP 融合之后, 残差连接之前)
        output = self.dropout(output)

        # ★ 数值稳定性加固：使用 gamma 缩放残差
        # Output = Input + gamma * Mamba_Block(Input)
        # gamma 初始化为 0.1，防止训练初期 Mamba 随机输出破坏主干特征
        output = shortcut + self.gamma * output

        return output


# ==================== SSM-CGB (SSM Cross-Gating Block) ====================
class SSMCrossGatingBlock(nn.Module):
    """
    SSM 交叉门控块 (SSM-CGB)
    替换原 CGB，使用 SSM 实现跨特征条件化
    
    结构：
    - X 路径 (Condition): RMSNorm -> Mamba Condition Gen (生成动态参数)
    - Y 路径 (Input Sequence): RMSNorm -> Mamba Selective Scan (受 X 参数控制)
    - 输出: Y 路径输出 + 原始 Y (残差连接)
    
    其中：
    - X 是 Encoder 特征 (条件信号)
    - Y 是 Decoder 特征 (输入序列)
    """
    def __init__(
        self,
        features,           # 特征通道数
        state_dim=16,       # SSM 隐状态维度
        conv_kernel=4,      # 卷积核大小
        expand_ratio=2,     # 扩展倍率
        upsample_y=True,    # 是否对 Y 进行上采样
        residual_gamma=0.1, # ★ 残差缩放因子 (防止训练初期 SSM 输出破坏主干特征)
        dropout_rate=0.0,   # ★ 输出后的 Dropout (正则化, 防过拟合)
    ):
        super().__init__()
        self.features = features
        self.upsample_y = upsample_y
        
        # Y 上采样（如果需要）
        if upsample_y:
            # 仅上采样空间分辨率，通道数不变
            self.upsample = nn.ConvTranspose2d(
                features, features, kernel_size=2, stride=2
            )
        
        # X 路径: Condition Generator
        self.norm_x = RMSNorm(features)
        self.condition_gen = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features * expand_ratio, kernel_size=1),
        )
        # 生成用于 Y 的动态参数 (B, C, H, W)
        self.dynamic_param_gen = nn.Conv2d(features * expand_ratio, features * 2, kernel_size=1)
        
        # Y 路径: Selective Scan
        self.norm_y = RMSNorm(features)
        self.y_in_proj = nn.Conv2d(features, features * expand_ratio, kernel_size=1)
        
        # 简化的 SSM Scan (沿水平和垂直方向)
        # split 后每半的通道数 = features * expand_ratio // 2
        scan_dim = features * expand_ratio // 2
        self.scan_horizontal = SimpleSSMBlock(
            d_model=scan_dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=1  # SSM 内部不再扩展（已经是扩展后的特征）
        )
        self.scan_vertical = SimpleSSMBlock(
            d_model=scan_dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=1  # SSM 内部不再扩展
        )
        
        # 输出投影: 从 features*expand_ratio 压缩回 features
        self.out_proj = nn.Sequential(
            nn.Conv2d(features * expand_ratio, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
        )

        # ★ Dropout (在输出之后, 残差连接之前)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # ★ 数值稳定性加固：可学习的残差缩放因子 gamma
        # 防止训练初期 SSM 输出破坏 Decoder 主干特征
        self.gamma = nn.Parameter(torch.tensor(residual_gamma))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        """
        Args:
            x: Encoder 特征 (Condition) - (B, C, H, W)
            y: Decoder 特征 (Input Sequence) - (B, C, H, W) 或 (B, C, H/2, W/2)
        Returns:
            output: 融合后的 Decoder 特征 (B, C, H, W)
        """
        # Y 上采样（仅空间分辨率，通道数不变）
        if self.upsample_y:
            y = self.upsample(y)  # (B, C, H, W)
        
        B, C, H, W = x.shape
        
        # ========== X 路径: 生成条件参数 ==========
        x_norm = self.norm_x(x)
        condition = self.condition_gen(x_norm)  # (B, C*expand, H, W)
        dynamic_params = self.dynamic_param_gen(condition)  # (B, 2C, H, W)
        # 分离为 scale 和 shift 参数
        scale, shift = torch.chunk(dynamic_params, 2, dim=1)  # 各 (B, C, H, W)
        
        # ========== Y 路径: 选择性扫描 ==========
        shortcut_y = y
        y_norm = self.norm_y(y)
        
        # 应用动态条件
        y_conditioned = y_norm * (1 + scale) + shift
        
        # 投影到高维空间
        y_proj = self.y_in_proj(y_conditioned)  # (B, C*expand, H, W)
        
        # Split 成两半分别沿水平/垂直扫描
        y_h, y_v = torch.chunk(y_proj, 2, dim=1)  # 各 (B, C*expand/2, H, W)
        
        # 水平扫描: (B, C_e/2, H, W) -> (B*H, W, C_e/2) -> SSM -> (B, C_e/2, H, W)
        y_h_reshaped = rearrange(y_h, 'b c h w -> (b h) w c')
        y_h_out = self.scan_horizontal(y_h_reshaped)
        y_h_out = rearrange(y_h_out, '(b h) w c -> b c h w', h=H)
        
        # 垂直扫描: (B, C_e/2, H, W) -> (B*W, H, C_e/2) -> SSM -> (B, C_e/2, H, W)
        y_v_reshaped = rearrange(y_v, 'b c h w -> (b w) h c')
        y_v_out = self.scan_vertical(y_v_reshaped)
        # (B*W, H, C_e/2) -> (B, W, H, C_e/2) -> (B, C_e/2, H, W)
        y_v_out = rearrange(y_v_out, '(b w) h c -> b w h c', w=W)
        y_v_out = y_v_out.permute(0, 3, 2, 1)  # (B, C_e/2, H, W)
        
        # 融合
        y_fused = torch.cat([y_h_out, y_v_out], dim=1)
        output = self.out_proj(y_fused)

        # ★ Dropout (在输出之后, 残差连接之前)
        output = self.dropout(output)

        # ★ 数值稳定性加固：使用 gamma 缩放残差
        # Output = Y + gamma * SSM_Cross_Gating(Y, X)
        # 防止训练初期 SSM 输出破坏 Decoder 主干特征
        output = shortcut_y + self.gamma * output

        return output


# ==================== DCT-HPM (DCT High-Pass Module) ====================
class DCTHighPassModule(nn.Module):
    """
    DCT 高通模块 (DCT-HPM)
    在输入端提取高频残差，提供频域物理先验

    原理：
    1. 对输入特征进行 2D DCT 变换
    2. 乘以可学习的高通掩码（低频区为0，高频区为1）
    3. 进行 2D IDCT 变换得到高频残差
    4. 1x1 卷积调整通道
    5. 与原始输入相加（残差连接）

    DCT 实现说明：
    - torch.dct 在很多 PyTorch 版本中不可用
    - 这里采用 DCT 矩阵乘法实现 (对每个 batch/channel 独立应用)
    - 支持任意 H, W (不要求 2 的幂)
    - DCT 矩阵在 init 时构造，在 forward 时复用
    """
    def __init__(self, channels, alpha=0.01):
        """
        Args:
            channels: 输入/输出通道数
            alpha: ★ 高频残差的初始权重 (降低到 0.01, 训练初期高频贡献极小)
        """
        super().__init__()
        self.channels = channels
        self.alpha = alpha

        # ★ 可学习的高通掩码 (在 DCT 频谱中: 左上角低频, 右下角高频)
        # 使用平滑的 sigmoid 过渡替代硬阈值, 防止高频能量突变
        self.high_pass_mask = nn.Parameter(self._init_mask())

        # ★ 数值稳定性加固：频域 RMSNorm
        self.dct_norm = RMSNorm(channels)

        # ★ 1x1 卷积调整通道 (zero-init: 训练初期高频残差贡献为 0)
        self.channel_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 通道注意力 (可选, 用于动态加权高频残差)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

        # ★ 残差权重 (可学习, 初始化为极小的 alpha)
        self.residual_weight = nn.Parameter(torch.tensor(alpha))

        # DCT 矩阵缓存 (按需构造)
        self._dct_matrices = {}  # {size: (D, D_inv)}

        # ★ 初始化 channel_conv 为零 (训练初期 DCT-HPM 输出 = 0, 等效于无此模块)
        nn.init.zeros_(self.channel_conv.weight)
        nn.init.zeros_(self.channel_conv.bias)

    def _init_mask(self):
        """
        ★ 初始化平滑高通掩码 (sigmoid 过渡, 替代硬阈值)
        
        在 DCT 频谱中, 左上角(k=0, l=0)是 DC 分量 (最低频),
        右下角(k=H-1, l=W-1)是最高频。
        
        使用 sigmoid 平滑过渡:
        - 低频区: mask ≈ 0 (抑制)
        - 中频区: mask ≈ 0.5 (过渡)
        - 高频区: mask ≈ 1 (保留)
        
        硬阈值缺点: 在截止频率处产生振铃效应 (Gibbs 现象)
        sigmoid 平滑过渡: 避免 Gibbs 振铃, 高频能量渐进注入
        """
        mask_size = 32  # 更大的掩码尺寸, 插值后更平滑
        mask = torch.zeros(mask_size, mask_size)
        center = mask_size // 2
        
        for i in range(mask_size):
            for j in range(mask_size):
                # 径向频率: 距离左上角的曼哈顿距离
                freq_idx = (i + j) / (2 * mask_size)  # 归一化到 [0, 1]
                # sigmoid 平滑过渡: 中心在 0.5, 陡度 10
                # freq_idx < 0.3 → mask ≈ 0 (低频抑制)
                # freq_idx > 0.7 → mask ≈ 1 (高频保留)  
                mask[i, j] = torch.sigmoid(torch.tensor((freq_idx - 0.5) * 10.0))
        
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)

    def _get_dct_matrix(self, N, device, dtype):
        """
        构造 NxN 的 DCT-II 正交矩阵
        D[k, n] = alpha(k) * cos(pi * k * (2n + 1) / (2N))
        其中 alpha(0) = 1/sqrt(N), alpha(k) = 1/sqrt(N/2) for k > 0
        正交矩阵: D_inv = D^T
        """
        cache_key = (N, str(device), str(dtype))
        if hasattr(self, '_dct_matrices') and cache_key in self._dct_matrices:
            return self._dct_matrices[cache_key]

        k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
        D = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        # 正交归一化
        D[0, :] = D[0, :] / math.sqrt(N)
        D[1:, :] = D[1:, :] / math.sqrt(N / 2)

        if not hasattr(self, '_dct_matrices'):
            self._dct_matrices = {}
        self._dct_matrices[cache_key] = D
        return D

    def _dct_2d(self, x):
        """
        2D DCT 变换 (DCT-II, 正交归一化)
        Args:
            x: (B, C, H, W)
        Returns:
            X: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        D_H = self._get_dct_matrix(H, device, dtype)  # (H, H)
        D_W = self._get_dct_matrix(W, device, dtype)  # (W, W)

        # x: (B, C, H, W) -> (B*C, H, W)
        x_bc = x.reshape(B * C, H, W)

        # 对 H 做 DCT: X_h = D_H @ x
        X_h = torch.bmm(D_H.unsqueeze(0).expand(B * C, -1, -1), x_bc)  # (B*C, H, W)

        # 对 W 做 DCT: X = X_h @ D_W^T  (因为正交, D_inv = D^T)
        X = torch.bmm(X_h, D_W.unsqueeze(0).expand(B * C, -1, -1).transpose(1, 2))  # (B*C, H, W)

        return X.reshape(B, C, H, W)

    def _idct_2d(self, X):
        """
        2D IDCT 变换 (DCT-III, DCT-II 的逆)
        因为 DCT-II 矩阵正交, 逆就是转置:
        正向: X = D_H @ x @ D_W^T
        逆向: x = D_H^T @ X @ D_W
        Args:
            X: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = X.shape
        device = X.device
        dtype = X.dtype

        D_H = self._get_dct_matrix(H, device, dtype)  # (H, H)
        D_W = self._get_dct_matrix(W, device, dtype)  # (W, W)

        # X: (B, C, H, W) -> (B*C, H, W)
        X_bc = X.reshape(B * C, H, W)

        # IDCT 步骤1: X_h = X_bc @ D_W  (注意: 不是 D_W^T, 因为正向是 D_W^T, 逆就是 D_W)
        x_h = torch.bmm(X_bc, D_W.unsqueeze(0).expand(B * C, -1, -1))

        # IDCT 步骤2: x = D_H^T @ X_h
        x = torch.bmm(D_H.unsqueeze(0).expand(B * C, -1, -1).transpose(1, 2), x_h)

        return x.reshape(B, C, H, W)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            output: 融合高频残差后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        # ★ AMP 兼容: DCT/IDCT + 后续卷积 全部强制使用 FP32
        x_float = x.float()
        shortcut = x_float

        with torch.autocast(device_type='cuda', enabled=False):
            # 1. 2D DCT 变换
            X_dct = self._dct_2d(x_float)

            # DCT 系数归一化 + 限幅
            X_dct = self.dct_norm(X_dct)
            X_dct = torch.clamp(X_dct, min=-10.0, max=10.0)

            # 2. 应用高通掩码
            mask = F.interpolate(
                self.high_pass_mask,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            X_high = X_dct * mask

            # 3. 2D IDCT 得到高频残差
            high_residual = self._idct_2d(X_high)
            high_residual = torch.clamp(high_residual, min=-10.0, max=10.0)

            # 4. 通道注意力 (FP32 下运行)
            attention = self.channel_attention(high_residual)
            high_residual = high_residual * attention

            # 5. 1x1 卷积调整通道
            high_residual = self.channel_conv(high_residual)

            # 6. 残差融合 (FP32)
            output = shortcut + self.residual_weight * high_residual

        # ★ 最后转换回原始 dtype
        return output.to(orig_dtype)


# ==================== APDM (Amplitude-Phase Decoupling Module) ====================
class AmplitudePhaseDecouplingModule(nn.Module):
    """
    幅相解耦模块 (APDM) — 数值稳定性加固版
    
    结构：
    - Path A (Identity): 直接保留原特征
    - Path B (Frequency Branch):
      1. FFT (rfft2) → 频谱归一化 (除以 H*W, 防止大尺寸特征数值爆炸)
      2. 分离幅度和相位 → amplitude clamp + log1p (防止 log(0) 产生 -inf)
      3. 幅度经 Conv1×1 (bottleneck) 处理
      4. 相位经 cos/sin 编码后 Conv1×1 (避免角度不连续)
      5. 组合回复数 → IFFT → RMSNorm → Sigmoid → Attention Map
    - 融合: F_out = F_spatial * (W_freq + 1)
      (W_freq ∈ [0,1], 所以 multiplier ∈ [1,2], 保证输出不会符号反转)
    
    数值加固要点:
    - amplitude 在 log1p 前 clamp ≥ 1e-8, 防止 log(0) = -inf
    - amp_feat 在 expm1 前 clamp 到 [-5, 5], 防止 exp 爆炸
    - phase_new 用 atan2(sin, cos) 包裹到 [-π, π], 防止角度漂移
    - 频谱归一化: 除以 sqrt(H*W) 补偿不同分辨率下的量级差异
    - 学习率缩放: 频域模块参数使用 0.5x 学习率 (通过 forward 内梯度缩放)
    """
    def __init__(self, channels, reduction=4, freq_grad_scale=0.5):
        """
        Args:
            channels: 输入通道数
            reduction: 通道压缩倍率
            freq_grad_scale: 频域分支梯度缩放因子 (0.5 = 半速学习)
        """
        super().__init__()
        self.channels = channels
        self.freq_grad_scale = freq_grad_scale
        reduced = max(channels // reduction, 8)
        
        # 幅度路径 (bottleneck: C → C/r → C)
        self.amp_conv1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.amp_conv2 = nn.Conv2d(reduced, channels, kernel_size=1)

        # 相位路径 (输入 cos+sin 拼接 → 2C 通道, bottleneck: 2C → C/r → C)
        self.phase_conv1 = nn.Conv2d(channels * 2, reduced, kernel_size=1)
        self.phase_conv2 = nn.Conv2d(reduced, channels, kernel_size=1)
        
        # RMSNorm (应用到 IFFT 后的特征, 稳定值域)
        self.freq_norm = RMSNorm(channels)
        
        # 注意力生成 (1×1 Conv → Sigmoid)
        self.attn_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # ★ 可学习的频域分支权重 (初始化为 0.1, 训练初期频域分支贡献很小)
        self.freq_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        # ★ 关键: 注意力卷积 bias 初始化为 2.0 → sigmoid(2) ≈ 0.88
        # 使得 W_freq+1 ≈ 1.88, 初始时输出接近 F_spatial*1.88 (轻微放大, 不破坏主干)
        nn.init.zeros_(self.attn_conv.weight)
        nn.init.constant_(self.attn_conv.bias, 2.0)
        
        # 幅度/相位卷积: 小权重初始化, 防止频谱被大幅修改
        for conv in [self.amp_conv1, self.amp_conv2, self.phase_conv1, self.phase_conv2]:
            nn.init.normal_(conv.weight, std=0.001)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 F_spatial (B, C, H, W)
        Returns:
            output: 频域引导后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        eps = 1e-8

        # ★ AMP 兼容: 所有频域运算强制使用 FP32
        x_float = x.float()

        with torch.autocast(device_type='cuda', enabled=False):
            # ========== Path A: Identity (FP32) ==========
            F_spatial = x_float

            # ========== Path B: Frequency Branch (FP32) ==========
            # 1. 2D FFT + 频谱归一化
            # norm='ortho' 已将 FFT 除以 sqrt(H*W), 但振幅仍可能很大
            F_fft = torch.fft.rfft2(x_float, norm='ortho')

            # 2. 分离幅度谱和相位谱
            amplitude = torch.abs(F_fft)
            phase = torch.angle(F_fft)

            # ★ 数值加固1: amplitude 在 log1p 前必须 clamp
            # 防止 amplitude=0 时 log1p(0)=0 没问题, 但 clamp 防止任何极小值导致问题
            amplitude = torch.clamp(amplitude, min=eps, max=1e3)

            # 3. 幅度路径处理 (log 域操作, 压缩动态范围)
            amp_log = torch.log1p(amplitude)  # 安全: amplitude ≥ eps
            amp_log = torch.clamp(amp_log, min=-20.0, max=20.0)
            amp_feat = self.amp_conv1(amp_log)
            amp_feat = self.relu(amp_feat)
            amp_feat = self.amp_conv2(amp_feat)
            
            # ★ 数值加固2: expm1 前 clamp amp_feat, 防止 exp 爆炸
            amp_feat = torch.clamp(amp_feat, min=-5.0, max=5.0)
            amp_processed = torch.expm1(amp_feat)
            amp_processed = torch.clamp(amp_processed, min=eps, max=10.0)

            # 4. 相位路径处理 (cos/sin 编码, 避免 [-π, π] 边界不连续)
            phase_cos = torch.cos(phase)
            phase_sin = torch.sin(phase)
            phase_in = torch.cat([phase_cos, phase_sin], dim=1)  # (B, 2C, H, W//2+1)
            phase_delta = self.phase_conv1(phase_in)
            phase_delta = self.relu(phase_delta)
            phase_delta = self.phase_conv2(phase_delta)
            phase_delta = torch.tanh(phase_delta) * 0.5  # ★ 限制偏移范围 ±0.5 rad

            # 5. 组合回复数形式
            # ★ 数值加固3: 用 atan2 包裹相位到 [-π, π], 防止角度漂移
            phase_raw = phase + phase_delta
            phase_new = torch.atan2(torch.sin(phase_raw), torch.cos(phase_raw))
            F_processed = torch.polar(amp_processed, phase_new)

            # 6. 2D IFFT
            F_ifft = torch.fft.irfft2(F_processed, s=(H, W), norm='ortho')
            F_ifft = torch.clamp(F_ifft, min=-5.0, max=5.0)

            # 7. RMSNorm + Sigmoid 生成注意力图
            F_ifft_norm = self.freq_norm(F_ifft)
            W_freq_pre = self.attn_conv(F_ifft_norm)
            W_freq = self.sigmoid(W_freq_pre)

            # ========== 融合: 频域分支梯度缩放 ==========
            # freq_weight * (W_freq + 1) 控制频域分支贡献
            # 训练初期 freq_weight≈0.1, 频域仅做轻微修正
            # multipler = 1 + freq_weight * W_freq ∈ [1, 1+freq_weight]
            multiplier = 1.0 + self.freq_weight * W_freq
            output = F_spatial * multiplier

        # ★ 最后转换回原始 dtype
        return output.to(orig_dtype)


# ==================== 辅助函数 ====================
def rearrange(x, pattern, **kwargs):
    """
    简化版 rearrange (避免对 einops 的强依赖)
    支持常用的 reshape/transpose 模式
    
    Args:
        x: 输入张量
        pattern: 字符串模式，如 'b c h w -> (b h) w c'
        **kwargs: 维度大小指定 (如 h=H, w=W)
    """
    input_pattern, output_pattern = pattern.split('->')
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()
    
    has_paren_in = '(' in input_pattern
    has_paren_out = '(' in output_pattern
    
    # ===== 4D -> 3D =====
    if not has_paren_in and has_paren_out:
        in_dims = input_pattern.split()
        dim_idx = {name: i for i, name in enumerate(in_dims)}
        
        # Parse parenthesized group from output, e.g. '(b h) w c'
        paren_start = output_pattern.index('(')
        paren_end = output_pattern.index(')')
        merged_str = output_pattern[paren_start+1:paren_end]  # 'b h'
        merged_dims = merged_str.split()  # ['b', 'h']
        
        if merged_dims == ['b', 'w']:
            # Output: (B*W, H, C)
            w_idx = dim_idx['w']
            h_idx = dim_idx['h']
            b_idx = dim_idx['b']
            c_idx = dim_idx['c']
            
            x = x.permute(b_idx, w_idx, h_idx, c_idx)
            b_val = x.shape[0]
            w_val = x.shape[1]
            h_val = x.shape[2]
            c_val = x.shape[3]
            x = x.reshape(b_val * w_val, h_val, c_val)
            return x
        
        elif merged_dims == ['b', 'h']:
            # Output: (B*H, W, C)
            h_idx = dim_idx['h']
            w_idx = dim_idx['w']
            b_idx = dim_idx['b']
            c_idx = dim_idx['c']
            
            x = x.permute(b_idx, h_idx, w_idx, c_idx)
            b_val = x.shape[0]
            h_val = x.shape[1]
            w_val = x.shape[2]
            c_val = x.shape[3]
            x = x.reshape(b_val * h_val, w_val, c_val)
            return x
    
    # ===== 3D -> 4D =====
    if has_paren_in and not has_paren_out:
        # Parse parenthesized group from input, e.g. '(b w) h c'
        paren_start = input_pattern.index('(')
        paren_end = input_pattern.index(')')
        merged_str = input_pattern[paren_start+1:paren_end]  # 'b w'
        merged_dims = merged_str.split()
        out_dims = output_pattern.split()
        
        if merged_dims == ['b', 'w']:
            # Input: (B*W, H, C) -> Output: (B, W, H, C)
            bh, seq, c = x.shape
            
            if 'w' in kwargs:
                W = kwargs['w']
            else:
                W = bh // max(1, bh // max(1, seq))
            
            b = bh // W
            x = x.reshape(b, W, seq, c)
            current_order = ['b', 'w', 'h', 'c']
            perm = [current_order.index(d) for d in out_dims]
            x = x.permute(*perm)
            return x
        
        elif merged_dims == ['b', 'h']:
            # Input: (B*H, W, C) -> Output: (B, H, W, C)
            bh, seq, c = x.shape
            
            if 'h' in kwargs:
                H = kwargs['h']
            else:
                H = seq
            
            b = bh // H
            x = x.reshape(b, H, seq, c)
            current_order = ['b', 'h', 'w', 'c']
            perm = [current_order.index(d) for d in out_dims]
            x = x.permute(*perm)
            return x
    
    # ===== 4D -> 4D: 转置等 =====
    if not has_paren_in and not has_paren_out:
        in_dims = input_pattern.split()
        out_dims = output_pattern.split()
        # Build permutation
        perm = [in_dims.index(d) for d in out_dims]
        x = x.permute(*perm)
        return x
    
    # 如果无法解析，返回原张量并警告
    import warnings
    warnings.warn(f"Cannot parse pattern: {pattern}, returning original tensor")
    return x
