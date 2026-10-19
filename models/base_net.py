"""
基础多轴网络 - PyTorch实现
多阶段、多尺度的U-Net架构，核心是MAMLP和CGB模块

架构特点：
1. U-Net形状的层级结构，Encoder-Decoder-Bottleneck
2. 多阶段(Stage)堆叠，每个阶段都是一个完整的U-Net
3. 多尺度监督，每个阶段输出多个尺度的预测
4. 核心模块：MultiAxisGmlpBlock (MAB) 和 CrossGatingBlock (CGB)

对应基础多轴 MLP 网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 兼容性导入：不同PyTorch版本的checkpoint位置不同
try:
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
except ImportError:
    try:
        from torch.utils import checkpoint as torch_checkpoint
    except ImportError:
        # 如果都不存在，提供一个不使用checkpoint的回退
        def torch_checkpoint(func, *args, **kwargs):
            return func(*args, **kwargs)

from .layers import (
    Conv2dSame,
    ConvTranspose2dSame,
    LayerNorm2d,
    RCAB,
    UpSampleRatio,
    weight_init,
)
from .blocks import MultiAxisGmlpBlock, CrossGatingBlock, BottleneckBlock, BottleneckWithAPDM, SAM


# ==================== Encoder块 ====================


class UNetEncoderBlock(nn.Module):
    """
    基础网络的Encoder块
    包含：输入投影 -> 多组(MAB + RCAB) -> 可选下采样

    Encoder 块
    """

    def __init__(
        self,
        features,  # 当前层基础通道C
        block_size,  # MAB局部分块窗口尺寸
        grid_size,  # MAB全局网格划分尺寸
        num_groups=1,  # MAB+RCAB堆叠组数（论文默认2组）
        lrelu_slope=0.2,  # RCAB内LeakyReLU负斜率
        block_gmlp_factor=2,  # MAB局部分支通道扩张倍数
        grid_gmlp_factor=2,  # MAB全局分支通道扩张倍数
        input_proj_factor=2,
        channels_reduction=4,  # RCAB通道注意力压缩倍率
        dropout_rate=0.0,
        in_channels=None,  # 【新增】输入通道数，默认等于features
        downsample=True,  # 是否2倍下采样（前3层开启，瓶颈层关闭）
        use_global_mlp=True,  # 是否使用MAB全局多轴MLP
        use_bias=True,
        use_cross_gating=False,  # 多Stage场景是否开启CGB跨特征融合
    ):
        super().__init__()
        self.features = features
        self.num_groups = num_groups
        self.downsample = downsample
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.in_channels = in_channels if in_channels is not None else features

        # 1. 输入1×1 Same卷积：通道融合/平滑，skip拼接后用来压缩通道
        self.input_proj = Conv2dSame(
            self.in_channels, features, kernel_size=1, bias=use_bias
        )

        # 2. 存储多组MAB、RCAB
        self.mab_blocks = nn.ModuleList()
        self.rcab_blocks = nn.ModuleList()

        # 循环构建每组(MAB + RCAB)
        for i in range(num_groups):
            # 开启全局MLP才创建MAB
            if use_global_mlp:
                self.mab_blocks.append(
                    MultiAxisGmlpBlock(
                        block_size=block_size,
                        grid_size=grid_size,
                        block_gmlp_factor=block_gmlp_factor,
                        grid_gmlp_factor=grid_gmlp_factor,
                        input_proj_factor=input_proj_factor,
                        use_bias=use_bias,
                        dropout_rate=dropout_rate,
                    )
                )
            # RCAB必创建（Encoder标准卷积注意力块）
            self.rcab_blocks.append(
                RCAB(
                    features=features,
                    reduction=channels_reduction,
                    lrelu_slope=lrelu_slope,
                    use_bias=use_bias,
                )
            )

        # 3. 跨阶段CGB交叉门控（多Stage才实例化）
        if use_cross_gating:
            self.cross_gating = CrossGatingBlock(
                features=features,
                block_size=block_size,
                grid_size=grid_size,
                dropout_rate=dropout_rate,
                input_proj_factor=input_proj_factor,
                upsample_y=False,  # enc+dec特征分辨率和x一致，无需上采样
                use_bias=use_bias,
            )

        # 4. 下采样层：4×4 kernel stride=2，TF Same，H/W直接减半
        if downsample:
            self.down_conv = Conv2dSame(
                features, 2 * features, kernel_size=4, stride=2, bias=use_bias
            )

    def forward(self, x, skip=None, enc=None, dec=None):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            skip: 跳跃连接特征（多尺度输入）
            enc: 上一阶段encoder特征（跨阶段融合）
            dec: 上一阶段decoder特征（跨阶段融合）

        Returns:
            x_down: 下采样后的特征（如果downsample=True）
            x: 当前层特征（用于跳跃连接）
        """
        # 拼接skip连接
        if skip is not None:
            # 1. 通道拼接 x [B,C,H,W] + skip [B,C,H,W] → [B,2C,H,W]
            x = torch.cat([x, skip], dim=1)
            # 2. 1×1卷积压缩通道回 C
            x = self.input_proj(x)

        shortcut_long = x

        # 多组MAB + RCAB
        for i in range(self.num_groups):
            if self.use_global_mlp:  # MAB：并行全局/局部空间混合，尺寸通道不变
                x = self.mab_blocks[i](x)
            x = self.rcab_blocks[i](x)

        # 长残差连接
        x = x + shortcut_long

        # 跨阶段交叉门控
        if enc is not None and dec is not None:
            assert self.use_cross_gating
            x, _ = self.cross_gating(x, enc + dec)

        # 下采样
        if self.downsample:
            x_down = self.down_conv(x)
            return x_down, x
        else:
            return x


# ==================== Decoder块 ====================


class UNetDecoderBlock(nn.Module):
    """
    基础网络的Decoder块
    上采样 + EncoderBlock（无下采样）

    Decoder 块
    """

    def __init__(
        self,
        features,
        block_size,
        grid_size,
        num_groups=1,
        lrelu_slope=0.2,
        block_gmlp_factor=2,
        grid_gmlp_factor=2,
        input_proj_factor=2,
        channels_reduction=4,
        dropout_rate=0.0,
        use_global_mlp=True,
        use_bias=True,
        in_channels=None,  # ✅【修复4】支持动态传入拼接后的总通道数
    ):
        super().__init__()

        # 上采样转置卷积：输入2*features，输出features
        self.up_conv = ConvTranspose2dSame(
            2 * features, features, kernel_size=2, stride=2, bias=use_bias
        )

        # Encoder块（无下采样）
        in_ch = in_channels if in_channels is not None else 2 * features
        self.encoder_block = UNetEncoderBlock(
            features=features,
            num_groups=num_groups,
            lrelu_slope=lrelu_slope,
            in_channels=in_ch,  # ✅ 传入正确的拼接通道数
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=block_gmlp_factor,
            grid_gmlp_factor=grid_gmlp_factor,
            input_proj_factor=input_proj_factor,
            channels_reduction=channels_reduction,
            use_global_mlp=use_global_mlp,
            dropout_rate=dropout_rate,
            downsample=False,
            use_bias=use_bias,
        )

    def forward(self, x, bridge=None):
        """
        Args:
            x: 输入特征 (B, C, H/2, W/2)
            bridge: 跳跃连接特征 (B, C, H, W)

        Returns:
            x: 上采样并融合后的特征 (B, C, H, W)
        """
        x = self.up_conv(x)
        x = self.encoder_block(x, skip=bridge)
        return x


# ==================== 基础多轴网络 ====================


class BaseNet(nn.Module):
    """
    BaseNet: Multi-Axis MLP U-Net for Image Processing

    多阶段、多尺度的图像恢复网络
    - 每个Stage都是一个完整的U-Net
    - 多尺度输入输出，多Loss监督
    - 核心模块：MAB (Multi-Axis Gated MLP) 和 CGB (Cross-Gating Block)

    输出格式：
    outputs_all = [
        [stage1_scale1, stage1_scale2, stage1_scale3],
        [stage2_scale1, stage2_scale2, stage2_scale3],
        ...
    ]
    最终输出为 outputs_all[-1][-1]

    对应基础多轴 MLP 网络
    """

    def __init__(
        self,
        features=64,
        depth=3,
        num_stages=2,
        num_groups=1,
        use_bias=True,
        num_supervision_scales=1,
        lrelu_slope=0.2,
        use_global_mlp=True,
        use_cross_gating=True,
        high_res_stages=2,
        block_size_hr=(16, 16),
        block_size_lr=(8, 8),
        grid_size_hr=(16, 16),
        grid_size_lr=(8, 8),
        num_bottleneck_blocks=1,
        block_gmlp_factor=2,
        grid_gmlp_factor=2,
        input_proj_factor=2,
        channels_reduction=4,
        num_outputs=3,
        dropout_rate=0.0,
    ):
        super().__init__()

        self.features = features
        self.depth = depth
        self.num_stages = num_stages
        self.num_groups = num_groups
        self.use_bias = use_bias
        self.num_supervision_scales = num_supervision_scales
        self.lrelu_slope = lrelu_slope
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.high_res_stages = high_res_stages
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate

        self.encoder_last_dropout = nn.Dropout2d(0.05)
        self.decoder_first_dropout = nn.Dropout2d(0.05)

        # 构建各阶段的模块
        self._build_stages()

    def _get_block_grid_size(self, stage_idx):
        """根据stage索引获取block和grid大小"""
        if stage_idx < self.high_res_stages:
            return self.block_size_hr, self.grid_size_hr
        else:
            return self.block_size_lr, self.grid_size_lr

    def _build_stages(self):
        """构建所有阶段的模块"""
        # 输入卷积（每个stage每个scale一个）
        self.input_convs = nn.ModuleList()

        # 输入融合CGB（用于stage>0时融合SAM特征）
        if self.use_cross_gating:
            self.input_fuse_cgb = nn.ModuleList()
        else:
            self.input_cat_convs = nn.ModuleList()

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()

        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList()

        # Cross gating blocks (skip connection处)
        if self.use_cross_gating:
            self.cross_gating_blocks = nn.ModuleList()
            self.skip_proj_layers = (
                nn.ModuleList()
            )  # ✅【修复1】必须放在循环外，防止被覆盖
        else:
            self.skip_convs = nn.ModuleList()

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        # UpSampleRatio modules for Cross Gating (skip connection fusion)
        self.cross_gate_upsample = nn.ModuleList()
        
        # UpSampleRatio modules for Decoder (skip feature upsampling)
        self.decoder_upsample = nn.ModuleList()

        # SAM modules (非最后一个stage使用)
        if self.num_stages > 1:
            self.sam_modules = nn.ModuleList()

        # 输出卷积（最后一个stage使用）
        self.output_convs = nn.ModuleList()

        # 循环创建num_stages个 Stage
        for idx_stage in range(self.num_stages):
            # ===== 输入卷积 =====
            stage_input_convs = nn.ModuleList()
            for i in range(self.num_supervision_scales):
                stage_input_convs.append(
                    Conv2dSame(
                        3,  # 输入是RGB图像
                        (2**i) * self.features,
                        kernel_size=3,
                        bias=self.use_bias,
                    )
                )
            self.input_convs.append(stage_input_convs)

            # ===== 输入融合（跨阶段SAM特征融合） =====
            if idx_stage > 0:
                if self.use_cross_gating:
                    stage_input_fuse = nn.ModuleList()
                    for i in range(self.num_supervision_scales):
                        block_size, grid_size = self._get_block_grid_size(i)
                        stage_input_fuse.append(
                            CrossGatingBlock(
                                features=(2**i) * self.features,
                                block_size=block_size,
                                grid_size=grid_size,
                                dropout_rate=self.dropout_rate,
                                input_proj_factor=self.input_proj_factor,
                                upsample_y=False,
                                use_bias=self.use_bias,
                            )
                        )
                    self.input_fuse_cgb.append(stage_input_fuse)
                else:
                    stage_input_cat = nn.ModuleList()
                    for i in range(self.num_supervision_scales):
                        stage_input_cat.append(
                            Conv2dSame(
                                2 * (2**i) * self.features,
                                (2**i) * self.features,
                                kernel_size=1,
                                bias=self.use_bias,
                            )
                        )
                    self.input_cat_convs.append(stage_input_cat)

            # ===== Encoder blocks =====
            stage_encoders = nn.ModuleList()
            for i in range(self.depth):
                block_size, grid_size = self._get_block_grid_size(i)
                use_cross_gating_layer = idx_stage > 0
                feat = (2**i) * self.features
                # 有skip的层（多尺度输入）输入通道翻倍，无skip则和features一致
                in_ch = 2 * feat if i < self.num_supervision_scales else feat

                stage_encoders.append(
                    UNetEncoderBlock(
                        features=feat,
                        in_channels=in_ch,  # ✅ 传入对应输入通道
                        num_groups=self.num_groups,
                        downsample=True,
                        lrelu_slope=self.lrelu_slope,
                        block_size=block_size,
                        grid_size=grid_size,
                        block_gmlp_factor=self.block_gmlp_factor,
                        grid_gmlp_factor=self.grid_gmlp_factor,
                        input_proj_factor=self.input_proj_factor,
                        channels_reduction=self.channels_reduction,
                        use_global_mlp=self.use_global_mlp,
                        dropout_rate=self.dropout_rate,
                        use_bias=self.use_bias,
                        use_cross_gating=use_cross_gating_layer,
                    )
                )
            self.encoder_blocks.append(stage_encoders)

            # ===== Bottleneck blocks (带APDM并联分支) =====
            # 使用BottleneckWithAPDM替代原BottleneckBlock
            # 结构：分支A(原Bottleneck) + 分支B(APDM频域引导) -> Concat + Conv1x1融合
            stage_bottlenecks = nn.ModuleList()
            for i in range(self.num_bottleneck_blocks):
                stage_bottlenecks.append(
                    BottleneckWithAPDM(
                        block_size=self.block_size_lr,
                        grid_size=self.grid_size_lr,
                        # Encoder下采样depth次后通道为 2^depth * features
                        features=(2**self.depth) * self.features,
                        num_groups=self.num_groups,
                        block_gmlp_factor=self.block_gmlp_factor,
                        grid_gmlp_factor=self.grid_gmlp_factor,
                        input_proj_factor=self.input_proj_factor,
                        dropout_rate=self.dropout_rate,
                        use_bias=self.use_bias,
                        channels_reduction=self.channels_reduction,
                    )
                )
            self.bottleneck_blocks.append(stage_bottlenecks)

            # ===== Cross gating blocks (skip connections) =====
            if self.use_cross_gating:
                stage_cgb = nn.ModuleList()
                stage_skip_proj = nn.ModuleList()
                for i in range(self.depth):
                    block_size, grid_size = self._get_block_grid_size(i)
                    feat = (2**i) * self.features
                    # 多特征拼接后压缩到目标通道
                    total_ch = sum([(2**j) * self.features for j in range(self.depth)])
                    stage_skip_proj.append(
                        Conv2dSame(total_ch, feat, kernel_size=1, bias=self.use_bias)
                    )
                    stage_cgb.append(
                        CrossGatingBlock(
                            features=feat,
                            block_size=block_size,
                            grid_size=grid_size,
                            input_proj_factor=self.input_proj_factor,
                            dropout_rate=self.dropout_rate,
                            upsample_y=True,
                            use_bias=self.use_bias,
                        )
                    )
                self.cross_gating_blocks.append(stage_cgb)
                self.skip_proj_layers.append(
                    stage_skip_proj
                )  # ✅【修复3】只append，不重新初始化
            else:
                stage_skip = nn.ModuleList()
                for i in range(self.depth):
                    stage_skip.append(
                        nn.ModuleList(
                            [
                                Conv2dSame(
                                    sum(
                                        [
                                            (2**j) * self.features
                                            for j in range(self.depth)
                                        ]
                                    ),
                                    (2**i) * self.features,
                                    kernel_size=1,
                                    bias=self.use_bias,
                                ),
                                Conv2dSame(
                                    (2**i) * self.features,
                                    (2**i) * self.features,
                                    kernel_size=3,
                                    bias=self.use_bias,
                                ),
                            ]
                        )
                    )
                self.skip_convs.append(stage_skip)

            # ===== Decoder blocks =====
            stage_decoders = nn.ModuleList()
            # ✅【核心修复5】计算所有skip特征拼接后的总通道数
            total_skip_ch = sum([(2**j) * self.features for j in range(self.depth)])
            for i in range(self.depth):
                block_size, grid_size = self._get_block_grid_size(i)
                feat = (2**i) * self.features
                stage_decoders.append(
                    UNetDecoderBlock(
                        features=feat,
                        in_channels=feat + total_skip_ch,  # ✅ x通道 + 所有skip拼接通道
                        num_groups=self.num_groups,
                        lrelu_slope=self.lrelu_slope,
                        block_size=block_size,
                        grid_size=grid_size,
                        block_gmlp_factor=self.block_gmlp_factor,
                        grid_gmlp_factor=self.grid_gmlp_factor,
                        input_proj_factor=self.input_proj_factor,
                        channels_reduction=self.channels_reduction,
                        use_global_mlp=self.use_global_mlp,
                        dropout_rate=self.dropout_rate,
                        use_bias=self.use_bias,
                    )
                )
            self.decoder_blocks.append(stage_decoders)

            # ===== SAM / Output convs =====
            if idx_stage < self.num_stages - 1:
                # 非最后阶段，使用SAM
                stage_sams = nn.ModuleList()
                for i in range(self.num_supervision_scales):
                    if i < self.depth:
                        stage_sams.append(
                            SAM(
                                features=(2**i) * self.features,
                                output_channels=self.num_outputs,
                                use_bias=self.use_bias,
                            )
                        )
                self.sam_modules.append(stage_sams)
            else:
                # 最后阶段，使用输出卷积
                stage_outputs = nn.ModuleList()
                for i in range(self.num_supervision_scales):
                    if i < self.depth:
                        stage_outputs.append(
                            Conv2dSame(
                                (2**i) * self.features,
                                self.num_outputs,
                                kernel_size=3,
                                bias=self.use_bias,
                            )
                        )
                self.output_convs.append(stage_outputs)

            # ===== UpSampleRatio modules =====
            # Cross Gating upsample: for each i (depth), j (depth) pairs
            stage_cross_gate_upsample = nn.ModuleList()
            for i in range(self.depth):
                row = nn.ModuleList()
                for j in range(self.depth):
                    ch = (2**j) * self.features
                    ratio = 2 ** (j - i)
                    row.append(UpSampleRatio(ch, ratio=ratio, use_bias=self.use_bias))
                stage_cross_gate_upsample.append(row)
            self.cross_gate_upsample.append(stage_cross_gate_upsample)
            
            # Decoder upsample: for each i (depth), j (depth) pairs
            stage_decoder_upsample = nn.ModuleList()
            for i in range(self.depth):
                row = nn.ModuleList()
                for j in range(self.depth):
                    ch = (2**(self.depth - 1 - j)) * self.features
                    ratio = 2 ** (self.depth - j - 1 - i)
                    row.append(UpSampleRatio(ch, ratio=ratio, use_bias=self.use_bias))
                stage_decoder_upsample.append(row)
            self.decoder_upsample.append(stage_decoder_upsample)

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            outputs_all: 多阶段多尺度输出列表
        """
        b, c, h, w = x.shape

        # 生成多尺度输入图像
        shortcuts = [x]
        for i in range(1, self.num_supervision_scales):
            scale_h, scale_w = h // (2**i), w // (2**i)
            shortcuts.append(F.interpolate(x, size=(scale_h, scale_w), mode="nearest"))

        outputs_all = []
        sam_features = []
        encs_prev = []
        
        # 启用梯度检查点以节省显存
        # 支持手动启用（_use_gradient_checkpointing）或自动检测（batch_size > 4）
        use_checkpointing = getattr(self, '_use_gradient_checkpointing', False) or (x.shape[0] > 4)
        decs_prev = []
        
        def _encoder_forward(encoder_block, x, skip, enc, dec):
            return encoder_block(x, skip=skip, enc=enc, dec=dec)
        
        def _decoder_forward(decoder_block, x, bridge):
            return decoder_block(x, bridge=bridge)

        for idx_stage in range(self.num_stages):
            # ===== 多尺度输入特征 =====
            x_scales = []
            for i in range(self.num_supervision_scales):
                x_scale = self.input_convs[idx_stage][i](shortcuts[i])
                if idx_stage > 0:
                    sam_feat = sam_features.pop()
                    if self.use_cross_gating:
                        x_scale, _ = self.input_fuse_cgb[idx_stage - 1][i](
                            x_scale, sam_feat
                        )
                    else:
                        x_scale = torch.cat([x_scale, sam_feat], dim=1)
                        x_scale = self.input_cat_convs[idx_stage - 1][i](x_scale)
                x_scales.append(x_scale)

            # ===== Encoder =====
            encs = []
            x = x_scales[0]
            for i in range(self.depth):
                x_scale = x_scales[i] if i < self.num_supervision_scales else None
                enc_prev = encs_prev.pop() if idx_stage > 0 else None
                dec_prev = decs_prev.pop() if idx_stage > 0 else None

                if use_checkpointing and self.training:
                    x, bridge = torch_checkpoint(
                        _encoder_forward,
                        self.encoder_blocks[idx_stage][i],
                        x, x_scale, enc_prev, dec_prev,
                        use_reentrant=False
                    )
                else:
                    x, bridge = self.encoder_blocks[idx_stage][i](
                        x, skip=x_scale, enc=enc_prev, dec=dec_prev
                    )
                encs.append(bridge)

            # ===== Bottleneck =====
            for i in range(self.num_bottleneck_blocks):
                if use_checkpointing and self.training:
                    x = torch_checkpoint(
                        self.bottleneck_blocks[idx_stage][i], x,
                        use_reentrant=False
                    )
                else:
                    x = self.bottleneck_blocks[idx_stage][i](x)

            if self.training:
                x = self.encoder_last_dropout(x)

            global_feature = x

            # ===== Cross Gating (Skip connections) =====
            skip_features = []
            for i in reversed(range(self.depth)):
                signals = []
                for j, enc in enumerate(encs):
                    signals.append(
                        self.cross_gate_upsample[idx_stage][i][j](enc)
                    )
                signal = torch.cat(signals, dim=1)

                if self.use_cross_gating:
                    signal = self.skip_proj_layers[idx_stage][i](signal)  # ✅ 通道压缩
                    skips, global_feature = self.cross_gating_blocks[idx_stage][i](
                        signal, global_feature
                    )
                else:
                    skips = self.skip_convs[idx_stage][i][0](signal)
                    skips = self.skip_convs[idx_stage][i][1](skips)

                skip_features.append(skips)

            # ===== Decoder =====
            outputs = []
            decs = []
            sam_features = []

            for i in reversed(range(self.depth)):
                signals = []
                for j, skip in enumerate(skip_features):
                    signals.append(
                        self.decoder_upsample[idx_stage][i][j](skip)
                    )
                signal = torch.cat(signals, dim=1)

                if i == self.depth - 1 and self.training:
                    signal = self.decoder_first_dropout(signal)

                if use_checkpointing and self.training:
                    x = torch_checkpoint(
                        _decoder_forward,
                        self.decoder_blocks[idx_stage][i],
                        x, signal,
                        use_reentrant=False
                    )
                else:
                    x = self.decoder_blocks[idx_stage][i](x, bridge=signal)
                decs.append(x)

                if i < self.num_supervision_scales:
                    if idx_stage < self.num_stages - 1:
                        sam, output = self.sam_modules[idx_stage][i](x, shortcuts[i])
                        outputs.append(output)
                        sam_features.append(sam)
                    else:
                        output = self.output_convs[0][i](x)
                        output = output + shortcuts[i]
                        outputs.append(output)

            encs_prev = encs[::-1]
            decs_prev = decs
            # Reverse outputs to maintain scale order: [full_res, 1/2, 1/4, ...]
            # The decoder loop iterates from highest depth (smallest scale) to lowest
            # but we want to return outputs from largest scale to smallest
            # Note: sam_features is NOT reversed because it's accessed via pop()
            # which naturally returns elements in reverse order in the next stage
            outputs_all.append(outputs[::-1])

        return outputs_all
    
    def gradient_checkpointing_enable(self):
        """启用梯度检查点以节省显存"""
        self._use_gradient_checkpointing = True
        print("  [Gradient Checkpointing] Enabled for BaseNet model")
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self._use_gradient_checkpointing = False

    # ==================== 模型配置工厂函数 ====================


def create_base_net(variant=None, **kwargs):
    """
    创建基础网络模型的工厂函数

    Args:
        variant: 模型变体，可选:
            'S-1', 'S-2', 'S-3' (Small)
            'M-1', 'M-2', 'M-3' (Medium)
        **kwargs: 其他配置参数

    Returns:
        基础网络模型实例
    """
    configs = {
        # Small variants
        "S-1": {
            "features": 32,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        "S-2": {
            "features": 32,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        "S-3": {
            "features": 32,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # Medium variants
        "M-1": {
            "features": 64,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        "M-2": {
            "features": 64,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        "M-3": {
            "features": 64,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
    }

    if variant is not None:
        if variant not in configs:
            raise ValueError(
                f"Unknown variant: {variant}. Available: {list(configs.keys())}"
            )
        config = configs[variant].copy()
        config.update(kwargs)
    else:
        config = kwargs

    return BaseNet(**config)
