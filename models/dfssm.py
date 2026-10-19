"""
DF-SSM - 改进版多轴状态空间模型
集成四大核心新模块：
1. DCT-HPM: DCT高通物理先验 (输入端)
2. MASSM: 多轴状态空间混合器 (替换MAB)
3. SSM-CGB: SSM交叉门控块 (替换CGB)
4. APDM: 幅相解耦模块 (Bottleneck)

架构特点：
- 基于基础多轴网络的 U-Net 骨架
- 使用 SSM (状态空间模型) 替代 MLP 进行空间混合
- 引入频域物理先验 (DCT + FFT)
- 所有 Mamba 相关模块使用 RMSNorm 替代 LayerNorm

★ Config-Driven 设计 (支持消融实验):
- use_dct_hpm: 是否启用 DCT-HPM 模块
- use_massm: True 用 MASSM, False 回退到原 MAB (MultiAxisGmlpBlock)
- use_ssm_cross_gating: True 用 SSM-CGB, False 回退到原 CGB (CrossGatingBlock)
- use_apdm: 是否启用 APDM 模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入基础模块
from .layers import (
    Conv2dSame,
    ConvTranspose2dSame,
    RCAB,
    UpSampleRatio,
    weight_init,
)
from .blocks import SAM, MultiAxisGmlpBlock, CrossGatingBlock

# 导入新模块
from .new_modules import (
    MASSM,
    SSMCrossGatingBlock,
    DCTHighPassModule,
    AmplitudePhaseDecouplingModule,
)


# ==================== 默认配置 (用于消融实验) ====================
DEFAULT_CONFIG = {
    'use_dct_hpm': True,           # 启用 DCT 高通模块
    'use_massm': True,             # True=MASSM, False=原 MAB
    'use_ssm_cross_gating': True,  # True=SSM-CGB, False=原 CGB
    'use_apdm': True,              # 启用 APDM 频域引导
}


# ==================== 改进版 Encoder Block ====================
class ImprovedEncoderBlock(nn.Module):
    """
    改进版 Encoder Block (Config-Driven)
    - use_massm=True: 使用 MASSM (多轴状态空间混合器)
    - use_massm=False: 回退到原 MAB (MultiAxisGmlpBlock)
    - use_cross_gating=True: 使用 SSM-CGB
    - use_cross_gating=False: 不使用交叉门控 (原版在 forward 中单独处理)
    - use_dct_hpm=True: 在输入端添加 DCT-HPM
    """
    def __init__(
        self,
        features,           # 当前层基础通道 C
        state_dim=16,       # SSM 隐状态维度
        conv_kernel=4,      # 卷积核大小
        expand_ratio=2,     # SSM 扩展倍率
        num_groups=1,       # MASSM + RCAB 堆叠组数
        lrelu_slope=0.2,
        channels_reduction=4,
        dropout_rate=0.0,
        in_channels=None,
        downsample=True,
        use_cross_gating=False,  # 是否使用 SSM-CGB
        use_dct_hpm=False,       # 是否使用 DCT-HPM (仅第一个 stage)
        use_massm=True,          # ★ True=MASSM, False=原 MAB
        block_size=(8, 8),       # 原 MAB 的 block_size
        grid_size=(8, 8),        # 原 MAB 的 grid_size
    ):
        super().__init__()
        self.features = features
        self.num_groups = num_groups
        self.downsample = downsample
        self.use_cross_gating = use_cross_gating
        self.use_dct_hpm = use_dct_hpm
        self.use_massm = use_massm
        self.in_channels = in_channels if in_channels is not None else features
        self.dropout_rate = dropout_rate

        # ========== DCT-HPM (仅第一个 stage 的第一个 block) ==========
        if use_dct_hpm:
            self.dct_hpm = DCTHighPassModule(features)

        # ========== 输入投影 ==========
        self.input_proj = Conv2dSame(
            self.in_channels, features, kernel_size=1
        )

        # ========== 空间混合模块: MASSM 或 原 MAB ==========
        self.mixer_blocks = nn.ModuleList()
        self.rcab_blocks = nn.ModuleList()

        for i in range(num_groups):
            if use_massm:
                # 使用 MASSM (多轴状态空间混合器)
                self.mixer_blocks.append(
                    MASSM(
                        features=features,
                        state_dim=state_dim,
                        conv_kernel=conv_kernel,
                        expand_ratio=expand_ratio,
                        dropout_rate=dropout_rate,  # ★ 正则化
                    )
                )
            else:
                # 回退到原 MAB (MultiAxisGmlpBlock)
                self.mixer_blocks.append(
                    MultiAxisGmlpBlock(
                        block_size=block_size,
                        grid_size=grid_size,
                        block_gmlp_factor=expand_ratio,
                        grid_gmlp_factor=expand_ratio,
                        input_proj_factor=2,
                        use_bias=True,
                        dropout_rate=dropout_rate,
                    )
                )
            self.rcab_blocks.append(
                RCAB(
                    features=features,
                    reduction=channels_reduction,
                    lrelu_slope=lrelu_slope,
                )
            )

        # ========== SSM-CGB (替换原 CGB) ==========
        if use_cross_gating:
            self.ssm_cross_gating = SSMCrossGatingBlock(
                features=features,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                expand_ratio=expand_ratio,
                upsample_y=False,  # encoder 特征分辨率一致
                dropout_rate=dropout_rate,  # ★ 正则化
            )

        # ========== 下采样层 ==========
        if downsample:
            self.down_conv = Conv2dSame(
                features, 2 * features, kernel_size=4, stride=2
            )

    def forward(self, x, skip=None, enc=None, dec=None):
        """
        Args:
            x: 输入特征 (B, C, H, W)
            skip: 跳跃连接特征
            enc: encoder 特征 (用于 SSM-CGB)
            dec: decoder 特征 (用于 SSM-CGB)
        Returns:
            x_down: 下采样后的特征
            x: 当前层特征
        """
        # 1. DCT-HPM 预处理 (如果启用) - 训练和推理都启用
        if self.use_dct_hpm:
            x = self.dct_hpm(x)

        # 2. 拼接 skip 连接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.input_proj(x)

        shortcut = x

        # 3. 空间混合 (MASSM 或 MAB) + RCAB 堆叠
        for i in range(self.num_groups):
            x = self.mixer_blocks[i](x)
            x = self.rcab_blocks[i](x)

        # 4. 长残差连接
        x = x + shortcut

        # 5. SSM-CGB 交叉门控
        if enc is not None and dec is not None and self.use_cross_gating:
            # enc + dec 作为条件，x 作为输入序列
            x = self.ssm_cross_gating(x, enc + dec)

        # 6. 下采样
        if self.downsample:
            x_down = self.down_conv(x)
            return x_down, x
        else:
            return x


# ==================== 改进版 Decoder Block ====================
class ImprovedDecoderBlock(nn.Module):
    """
    改进版 Decoder Block
    - 上采样 + ImprovedEncoderBlock
    """
    def __init__(
        self,
        features,
        state_dim=16,
        conv_kernel=4,
        expand_ratio=2,
        num_groups=1,
        lrelu_slope=0.2,
        channels_reduction=4,
        dropout_rate=0.0,
        in_channels=None,
        use_massm=True,           # ★ 传递给内部 EncoderBlock
        block_size=(8, 8),
        grid_size=(8, 8),
    ):
        super().__init__()

        # 上采样转置卷积
        self.up_conv = ConvTranspose2dSame(
            2 * features, features, kernel_size=2, stride=2
        )

        # Encoder block (无下采样)
        in_ch = in_channels if in_channels is not None else 2 * features
        self.encoder_block = ImprovedEncoderBlock(
            features=features,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand_ratio=expand_ratio,
            num_groups=num_groups,
            lrelu_slope=lrelu_slope,
            channels_reduction=channels_reduction,
            in_channels=in_ch,
            downsample=False,
            use_cross_gating=False,  # decoder 内部不需要 CGB
            use_dct_hpm=False,
            use_massm=use_massm,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,  # ★ 正则化
        )
    
    def forward(self, x, bridge=None):
        """
        Args:
            x: 输入特征 (B, C, H/2, W/2)
            bridge: 跳跃连接特征 (B, C, H, W)
        Returns:
            x: 上采样并融合后的特征
        """
        x = self.up_conv(x)
        x = self.encoder_block(x, skip=bridge)
        return x


# ==================== 改进版 DF-SSM ====================
class DFSSM(nn.Module):
    """
    DF-SSM: 多轴状态空间模型 + 双频域物理先验
    
    与基础网络 的主要区别：
    1. Stage 1 输入端添加 DCT-HPM 模块
    2. 所有 MAB 替换为 MASSM (多轴状态空间混合器)
    3. 所有 CGB 替换为 SSM-CGB (SSM 交叉门控块)
    4. Bottleneck 位置串联 APDM (幅相解耦模块)
    5. 所有 LayerNorm 替换为 RMSNorm
    
    输出格式与基础网络 兼容：
    outputs_all = [
        [stage1_scale1, stage1_scale2, stage1_scale3],
        [stage2_scale1, stage2_scale2, stage2_scale3],
        ...
    ]
    """
    
    def __init__(
        self,
        features=64,
        depth=3,
        num_stages=2,
        num_groups=2,
        use_bias=True,
        num_supervision_scales=3,
        lrelu_slope=0.2,
        high_res_stages=2,
        block_size_hr=(16, 16),
        block_size_lr=(8, 8),
        grid_size_hr=(16, 16),
        grid_size_lr=(8, 8),
        num_bottleneck_blocks=2,
        block_gmlp_factor=2,
        grid_gmlp_factor=2,
        input_proj_factor=2,
        channels_reduction=4,
        num_outputs=3,
        dropout_rate=0.0,
        # 新模块参数
        ssm_state_dim=16,       # SSM 隐状态维度
        ssm_conv_kernel=4,      # SSM 卷积核
        ssm_expand_ratio=2,     # SSM 扩展倍率
        # ★ Config-Driven 消融实验开关
        use_dct_hpm=True,       # 是否启用 DCT-HPM
        use_massm=True,         # True=MASSM, False=原 MAB
        use_ssm_cross_gating=True,  # True=SSM-CGB, False=原 CGB
        use_apdm=True,          # 是否启用 APDM
        config=None,            # ★ 可选: 配置字典 (覆盖以上 4 个开关)
    ):
        super().__init__()

        # ★ Config-Driven: 如果传入 config 字典，则覆盖对应开关
        if config is not None:
            use_dct_hpm = config.get('use_dct_hpm', use_dct_hpm)
            use_massm = config.get('use_massm', use_massm)
            use_ssm_cross_gating = config.get('use_ssm_cross_gating', use_ssm_cross_gating)
            use_apdm = config.get('use_apdm', use_apdm)

        self.features = features
        self.depth = depth
        self.num_stages = num_stages
        self.num_groups = num_groups
        self.use_bias = use_bias
        self.num_supervision_scales = num_supervision_scales
        self.lrelu_slope = lrelu_slope
        self.high_res_stages = high_res_stages
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.channels_reduction = channels_reduction
        self.num_outputs = num_outputs

        # 新模块参数
        self.ssm_state_dim = ssm_state_dim
        self.ssm_conv_kernel = ssm_conv_kernel
        self.ssm_expand_ratio = ssm_expand_ratio
        self.dropout_rate = dropout_rate  # ★ 正则化: Dropout
        # ★ 保存消融实验配置
        self.use_dct_hpm = use_dct_hpm
        self.use_massm = use_massm
        self.use_ssm_cross_gating = use_ssm_cross_gating
        self.use_apdm = use_apdm

        # 原基础网络 用的 block/grid size (用于回退到 MAB)
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr

        # Dropout
        self.encoder_last_dropout = nn.Dropout2d(0.05)
        self.decoder_first_dropout = nn.Dropout2d(0.05)

        # 构建各阶段
        self._build_stages()
    
    def _build_stages(self):
        """构建所有阶段的模块"""
        # 输入卷积 (每个 stage 每个 scale 一个)
        self.input_convs = nn.ModuleList()
        
        # 输入融合 (SSM-CGB, 用于 stage>0)
        self.input_fuse_ssm_cgb = nn.ModuleList()
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        
        # Bottleneck blocks (MASSM + APDM)
        self.bottleneck_massm = nn.ModuleList()
        self.bottleneck_apdm = nn.ModuleList()
        
        # Cross gating blocks (SSM-CGB)
        self.cross_gating_blocks = nn.ModuleList()
        self.skip_proj_layers = nn.ModuleList()
        self.gf_proj_layers = nn.ModuleList()
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder upsample
        self.decoder_upsample = nn.ModuleList()
        self.cross_gate_upsample = nn.ModuleList()
        
        # SAM modules (非最后一个 stage)
        if self.num_stages > 1:
            self.sam_modules = nn.ModuleList()
        
        # 输出卷积 (最后一个 stage)
        self.output_convs = nn.ModuleList()
        
        # ========== 循环构建 num_stages 个 Stage ==========
        for idx_stage in range(self.num_stages):
            feat = (2 ** 0) * self.features  # 当前层通道数
            
            # ===== 输入卷积 =====
            stage_input_convs = nn.ModuleList()
            for i in range(self.num_supervision_scales):
                stage_input_convs.append(
                    Conv2dSame(3, (2**i) * self.features, kernel_size=3)
                )
            self.input_convs.append(stage_input_convs)
            
            # ===== 输入融合 (SSM-CGB 或 原 CGB) =====
            if idx_stage > 0:
                stage_input_fuse = nn.ModuleList()
                for i in range(self.num_supervision_scales):
                    if self.use_ssm_cross_gating:
                        # 使用 SSM-CGB
                        stage_input_fuse.append(
                            SSMCrossGatingBlock(
                                features=(2**i) * self.features,
                                state_dim=self.ssm_state_dim,
                                conv_kernel=self.ssm_conv_kernel,
                                expand_ratio=self.ssm_expand_ratio,
                                upsample_y=False,
                                dropout_rate=self.dropout_rate,  # ★ 正则化
                            )
                        )
                    else:
                        # 回退到原 CGB (使用较小的 block_size/grid_size)
                        stage_input_fuse.append(
                            CrossGatingBlock(
                                features=(2**i) * self.features,
                                block_size=self.block_size_lr,
                                grid_size=self.grid_size_lr,
                                upsample_y=False,
                                use_bias=self.use_bias,
                            )
                        )
                self.input_fuse_ssm_cgb.append(stage_input_fuse)

            # ===== Encoder blocks =====
            stage_encoders = nn.ModuleList()
            for i in range(self.depth):
                feat_i = (2**i) * self.features
                in_ch = 2 * feat_i if i < self.num_supervision_scales else feat_i

                # 第一个 stage 的第一个 encoder block 使用 DCT-HPM
                use_dct = (idx_stage == 0 and i == 0 and self.use_dct_hpm)

                # 根据 stage 选择 block_size/grid_size (高层用小尺寸避免 OOM)
                block_size_i = self.block_size_hr if i == 0 else self.block_size_lr
                grid_size_i = self.grid_size_hr if i == 0 else self.grid_size_lr

                encoder_block = ImprovedEncoderBlock(
                    features=feat_i,
                    state_dim=self.ssm_state_dim,
                    conv_kernel=self.ssm_conv_kernel,
                    expand_ratio=self.ssm_expand_ratio,
                    num_groups=self.num_groups,
                    lrelu_slope=self.lrelu_slope,
                    channels_reduction=self.channels_reduction,
                    in_channels=in_ch,
                    downsample=True,  # 所有 encoder 层都下采样（与基础网络 一致）
                    use_cross_gating=(idx_stage > 0) and self.use_ssm_cross_gating,
                    use_dct_hpm=use_dct,
                    use_massm=self.use_massm,
                    block_size=block_size_i,
                    grid_size=grid_size_i,
                    dropout_rate=self.dropout_rate,  # ★ 正则化
                )
                stage_encoders.append(encoder_block)
            self.encoder_blocks.append(stage_encoders)

            # ===== Bottleneck (MASSM 或 MAB + APDM) =====
            stage_massm = nn.ModuleList()
            stage_apdm = nn.ModuleList()
            for i in range(self.num_bottleneck_blocks):
                # 所有 encoder 层都下采样，所以 bottleneck 通道数 = 2^depth * features
                feat_b = (2 ** self.depth) * self.features

                # 空间混合模块: MASSM 或 MAB
                if self.use_massm:
                    mixer_block = MASSM(
                        features=feat_b,
                        state_dim=self.ssm_state_dim,
                        conv_kernel=self.ssm_conv_kernel,
                        expand_ratio=self.ssm_expand_ratio,
                        dropout_rate=self.dropout_rate,  # ★ 正则化
                    )
                else:
                    mixer_block = MultiAxisGmlpBlock(
                        block_size=self.block_size_lr,
                        grid_size=self.grid_size_lr,
                        block_gmlp_factor=self.ssm_expand_ratio,
                        grid_gmlp_factor=self.ssm_expand_ratio,
                        input_proj_factor=2,
                        use_bias=self.use_bias,
                        dropout_rate=0.0,
                    )
                stage_massm.append(mixer_block)

                # APDM block (如果启用)
                if self.use_apdm:
                    apdm_block = AmplitudePhaseDecouplingModule(
                        channels=feat_b,
                        reduction=self.channels_reduction,
                    )
                    stage_apdm.append(apdm_block)

            self.bottleneck_massm.append(stage_massm)
            self.bottleneck_apdm.append(stage_apdm)

            # ===== Cross Gating (SSM-CGB 或 原 CGB) =====
            stage_cross_gating = nn.ModuleList()
            stage_skip_proj = nn.ModuleList()
            stage_gf_proj = nn.ModuleList()  # global_feature 投影层
            for i in reversed(range(self.depth)):
                feat_i = (2**i) * self.features
                # bottleneck 通道数 = 2^depth * features（因为所有 encoder 都下采样）
                bottleneck_channels = (2**self.depth) * self.features

                # Skip projection (压缩通道)
                total_enc_channels = sum(
                    (2**k) * self.features for k in range(self.depth)
                )
                skip_proj = nn.Sequential(
                    nn.Conv2d(total_enc_channels, feat_i, kernel_size=1),
                )
                stage_skip_proj.append(skip_proj)

                # global_feature 投影层 (bottleneck_channels -> feat_i)
                if bottleneck_channels != feat_i:
                    gf_proj_layer = nn.Conv2d(
                        bottleneck_channels, feat_i, kernel_size=1
                    )
                else:
                    gf_proj_layer = nn.Identity()
                stage_gf_proj.append(gf_proj_layer)

                # 交叉门控模块: SSM-CGB 或 原 CGB
                if self.use_ssm_cross_gating:
                    cgb_block = SSMCrossGatingBlock(
                        features=feat_i,
                        state_dim=self.ssm_state_dim,
                        conv_kernel=self.ssm_conv_kernel,
                        expand_ratio=self.ssm_expand_ratio,
                        upsample_y=False,  # 分辨率已在 forward 中通过 F.interpolate 对齐
                        dropout_rate=self.dropout_rate,  # ★ 正则化
                    )
                else:
                    cgb_block = CrossGatingBlock(
                        features=feat_i,
                        block_size=self.block_size_lr,
                        grid_size=self.grid_size_lr,
                        upsample_y=False,
                        use_bias=self.use_bias,
                    )
                stage_cross_gating.append(cgb_block)

            self.cross_gating_blocks.append(stage_cross_gating)
            self.skip_proj_layers.append(stage_skip_proj)
            self.gf_proj_layers.append(stage_gf_proj)

            # ===== Decoder blocks =====
            stage_decoders = nn.ModuleList()
            # 计算所有 skip 特征拼接后的总通道数（与基础网络 一致）
            total_skip_ch = sum([(2**j) * self.features for j in range(self.depth)])
            for i in reversed(range(self.depth)):
                feat_i = (2**i) * self.features

                block_size_i = self.block_size_hr if i == 0 else self.block_size_lr
                grid_size_i = self.grid_size_hr if i == 0 else self.grid_size_lr

                decoder_block = ImprovedDecoderBlock(
                    features=feat_i,
                    state_dim=self.ssm_state_dim,
                    conv_kernel=self.ssm_conv_kernel,
                    expand_ratio=self.ssm_expand_ratio,
                    num_groups=self.num_groups,
                    lrelu_slope=self.lrelu_slope,
                    channels_reduction=self.channels_reduction,
                    in_channels=feat_i + total_skip_ch,  # x通道 + 所有skip拼接通道
                    use_massm=self.use_massm,
                    block_size=block_size_i,
                    grid_size=grid_size_i,
                    dropout_rate=self.dropout_rate,  # ★ 正则化
                )
                stage_decoders.append(decoder_block)
            self.decoder_blocks.append(stage_decoders)
            
            # Decoder upsample (仅用于通道调整，空间分辨率在 forward 中处理)
            # 注意：内部列表顺序需要与 skip_features 顺序一致（deep → shallow）
            stage_decoder_upsample = nn.ModuleList()
            for i in reversed(range(self.depth)):
                upsample_list = nn.ModuleList()
                for k in range(self.depth):
                    # skip_features[k] 的通道数 = (2^(depth-1-k)) * features
                    skip_channels = (2**(self.depth - 1 - k)) * self.features
                    upsample_list.append(
                        nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
                    )
                stage_decoder_upsample.append(upsample_list)
            self.decoder_upsample.append(stage_decoder_upsample)
            
            # Cross gate upsample (仅用于通道调整，空间分辨率在 forward 中处理)
            stage_cross_gate_upsample = nn.ModuleList()
            for i in reversed(range(self.depth)):
                upsample_list = nn.ModuleList()
                for j in range(self.depth):
                    enc_channels = (2**j) * self.features
                    upsample_list.append(
                        nn.Conv2d(enc_channels, enc_channels, kernel_size=1)
                    )
                stage_cross_gate_upsample.append(upsample_list)
            self.cross_gate_upsample.append(stage_cross_gate_upsample)
            
            # ===== SAM modules (非最后一个 stage) =====
            if idx_stage < self.num_stages - 1:
                stage_sam = nn.ModuleList()
                for i in range(self.num_supervision_scales):
                    sam_module = SAM(
                        features=(2**i) * self.features,
                        output_channels=self.num_outputs,
                    )
                    stage_sam.append(sam_module)
                if hasattr(self, 'sam_modules'):
                    self.sam_modules.append(stage_sam)
            
            # ===== 输出卷积 (最后一个 stage) =====
            if idx_stage == self.num_stages - 1:
                stage_output_convs = nn.ModuleList()
                for i in range(self.num_supervision_scales):
                    stage_output_convs.append(
                        Conv2dSame((2**i) * self.features, self.num_outputs, kernel_size=3)
                    )
                self.output_convs.append(stage_output_convs)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            outputs_all: 多阶段多尺度输出列表
        """
        b, c, h, w = x.shape

        # ★ 应用 cross gating (统一处理 SSM-CGB 和原版 CGB 的返回值差异)
        # SSM-CGB 返回单个 tensor; 原版 CGB 返回 tuple (x_out, y_out)
        # 为保持接口一致, 原版 CGB 取 x_out (聚合了 x, y 双方信号, 适合作为 skip feature)
        
        # 生成多尺度输入图像
        shortcuts = [x]
        for i in range(1, self.num_supervision_scales):
            scale_h, scale_w = h // (2**i), w // (2**i)
            shortcuts.append(
                F.interpolate(x, size=(scale_h, scale_w), mode="nearest")
            )
        
        outputs_all = []
        sam_features = []
        encs_prev = []
        decs_prev = []
        
        for idx_stage in range(self.num_stages):
            # ===== 多尺度输入特征 =====
            x_scales = []
            for i in range(self.num_supervision_scales):
                x_scale = self.input_convs[idx_stage][i](shortcuts[i])
                
                # 跨阶段输入融合 (使用 SSM-CGB 或原版 CGB)
                if idx_stage > 0:
                    sam_feat = sam_features.pop()
                    fuse_result = self.input_fuse_ssm_cgb[idx_stage - 1][i](
                        x_scale, sam_feat
                    )
                    # ★ 兼容原版 CGB 的 tuple 返回值 (x_out, y_out)
                    if isinstance(fuse_result, tuple):
                        fuse_result = fuse_result[0]  # 取 x_out (主特征)
                    x_scale = fuse_result
                
                x_scales.append(x_scale)
            
            # ===== Encoder =====
            encs = []
            x = x_scales[0]
            for i in range(self.depth):
                x_scale = x_scales[i] if i < self.num_supervision_scales else None
                enc_prev = encs_prev.pop() if idx_stage > 0 else None
                dec_prev = decs_prev.pop() if idx_stage > 0 else None
                
                enc_out = self.encoder_blocks[idx_stage][i](
                    x, skip=x_scale, enc=enc_prev, dec=dec_prev
                )
                # 根据是否下采样，返回值不同
                if isinstance(enc_out, tuple):
                    x, bridge = enc_out
                else:
                    bridge = enc_out
                encs.append(bridge)
            
            # ===== Bottleneck (MASSM + APDM) =====
            for i in range(len(self.bottleneck_massm[idx_stage])):
                # MASSM 处理
                x = self.bottleneck_massm[idx_stage][i](x)

                # APDM 频域引导
                if self.use_apdm and i < len(self.bottleneck_apdm[idx_stage]):
                    x = self.bottleneck_apdm[idx_stage][i](x)
            
            if self.training:
                x = self.encoder_last_dropout(x)
            
            global_feature = x
            
            # ===== Cross Gating (SSM-CGB) =====
            skip_features = []
            cg_idx = 0  # cross_gating 索引（存储顺序与构建时一致: depth-1, ..., 0）
            for i in reversed(range(self.depth)):
                feat_i = (2**i) * self.features
                
                # 目标分辨率: 第 i 层 encoder 输出的分辨率
                target_h, target_w = encs[i].shape[2], encs[i].shape[3]
                
                # 收集所有 encoder 特征并调整到目标分辨率
                signals = []
                for j, enc in enumerate(encs):
                    # 通道调整 (1x1 conv)
                    enc_adjusted = self.cross_gate_upsample[idx_stage][cg_idx][j](enc)
                    # 空间分辨率对齐
                    if enc_adjusted.shape[2] != target_h or enc_adjusted.shape[3] != target_w:
                        enc_adjusted = F.interpolate(
                            enc_adjusted, size=(target_h, target_w),
                            mode='bilinear', align_corners=False
                        )
                    signals.append(enc_adjusted)
                
                signal = torch.cat(signals, dim=1)
                
                # 通道压缩
                signal = self.skip_proj_layers[idx_stage][cg_idx](signal)
                
                # 将 global_feature 上采样到目标分辨率并投影
                if global_feature.shape[2] != target_h or global_feature.shape[3] != target_w:
                    gf_upsampled = F.interpolate(
                        global_feature, size=(target_h, target_w),
                        mode='bilinear', align_corners=False
                    )
                else:
                    gf_upsampled = global_feature
                
                # 通道投影到 feat_i
                gf_projected = self.gf_proj_layers[idx_stage][cg_idx](gf_upsampled)
                
                # SSM-CGB 或原版 CGB: signal 作为 Condition (X), gf_projected 作为 Input (Y)
                cgb_result = self.cross_gating_blocks[idx_stage][cg_idx](
                    signal, gf_projected
                )
                # ★ 兼容原版 CGB 的 tuple 返回值 (x_out, y_out)
                # 取 x_out (聚合了 signal 和 gf_projected 双方信号, 适合作为 skip feature)
                if isinstance(cgb_result, tuple):
                    cgb_result = cgb_result[0]
                skip_out = cgb_result
                skip_features.append(skip_out)
                cg_idx += 1
            
            # ===== Decoder =====
            outputs = []
            decs = []
            
            # decoder_blocks 存储顺序是 [depth-1, ..., 0]（与构建时一致）
            decoder_idx = 0
            for i in reversed(range(self.depth)):
                feat_i = (2**i) * self.features
                target_h, target_w = encs[i].shape[2], encs[i].shape[3]
                
                # 收集 skip features 并调整到目标分辨率
                signals = []
                for j, skip in enumerate(skip_features):
                    skip_adjusted = self.decoder_upsample[idx_stage][decoder_idx][j](skip)
                    if skip_adjusted.shape[2] != target_h or skip_adjusted.shape[3] != target_w:
                        skip_adjusted = F.interpolate(
                            skip_adjusted, size=(target_h, target_w),
                            mode='bilinear', align_corners=False
                        )
                    signals.append(skip_adjusted)
                signal = torch.cat(signals, dim=1)
                
                if i == self.depth - 1 and self.training:
                    signal = self.decoder_first_dropout(signal)
                
                # Decoder block (使用 decoder_idx 因为存储顺序与构建时一致)
                x = self.decoder_blocks[idx_stage][decoder_idx](x, bridge=signal)
                decs.append(x)
                decoder_idx += 1
                
                if i < self.num_supervision_scales:
                    if idx_stage < self.num_stages - 1:
                        # SAM (监督注意力模块)
                        sam, output = self.sam_modules[idx_stage][i](
                            x, shortcuts[i]
                        )
                        outputs.append(output)
                        sam_features.append(sam)
                    else:
                        # 最后一个 stage: 输出卷积 + 残差
                        output = self.output_convs[0][i](x)
                        output = output + shortcuts[i]
                        outputs.append(output)
            
            encs_prev = encs[::-1]
            decs_prev = decs
            
            # 反转输出顺序: [full_res, 1/2, 1/4, ...]
            outputs_all.append(outputs[::-1])
        
        return outputs_all

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """向后兼容: APDM 改名 (原名 RFGM) 之前训练的 checkpoint, 其 bottleneck 键名前缀
        仍是 'bottleneck_rfgm.'。此处自动重映射到 'bottleneck_apdm.', 使旧权重可直接加载。
        """
        legacy_prefix = "bottleneck_rfgm."
        new_prefix = "bottleneck_apdm."
        if any(k.startswith(legacy_prefix) for k in state_dict):
            state_dict = {
                (new_prefix + k[len(legacy_prefix):] if k.startswith(legacy_prefix) else k): v
                for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        self._use_gradient_checkpointing = True
        print("  [Gradient Checkpointing] Enabled for DFSSM model")
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self._use_gradient_checkpointing = False


# ==================== 工厂函数 ====================
def create_dfssm(variant=None, config=None, **kwargs):
    """
    创建 DF-SSM 模型的工厂函数 (Config-Driven, 支持消融实验)

    Args:
        variant: 模型变体 (S-1, S-2, S-3, M-1, M-2, M-3)
        config: 消融实验配置字典，可选 keys:
            - use_dct_hpm (bool): 是否启用 DCT-HPM
            - use_massm (bool): True=MASSM, False=原 MAB
            - use_ssm_cross_gating (bool): True=SSM-CGB, False=原 CGB
            - use_apdm (bool): 是否启用 APDM
        **kwargs: 额外配置参数 (覆盖 variant 默认值)
    Returns:
        DFSSM 模型实例

    消融实验示例:
        # 完整模型 (默认)
        model = create_dfssm('M-2')

        # 关闭 APDM
        model = create_dfssm('M-2', config={'use_apdm': False})

        # 关闭 MASSM, 回退到原 MAB
        model = create_dfssm('M-2', config={'use_massm': False})

        # 原始基础网络 (所有新模块关闭)
        model = create_dfssm('M-2', config={
            'use_dct_hpm': False, 'use_massm': False,
            'use_ssm_cross_gating': False, 'use_apdm': False
        })
    """
    # 继承原版配置，添加新模块参数
    configs = {
        # Small variants
        "S-1": {
            "features": 32, "depth": 3, "num_stages": 1, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
        "S-2": {
            "features": 32, "depth": 3, "num_stages": 2, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
        "S-3": {
            "features": 32, "depth": 3, "num_stages": 3, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
        # Medium variants
        "M-1": {
            "features": 64, "depth": 3, "num_stages": 1, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
        "M-2": {
            "features": 64, "depth": 3, "num_stages": 2, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
        "M-3": {
            "features": 64, "depth": 3, "num_stages": 3, "num_groups": 2,
            "num_bottleneck_blocks": 2, "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2, "input_proj_factor": 2,
            "channels_reduction": 4, "num_supervision_scales": 3,
        },
    }

    if variant is not None:
        if variant not in configs:
            raise ValueError(
                f"Unknown variant: {variant}. Available: {list(configs.keys())}"
            )
        model_config = configs[variant].copy()
        # V2 特有参数 (默认全部启用)
        model_config.setdefault('ssm_state_dim', 16)
        model_config.setdefault('ssm_conv_kernel', 4)
        model_config.setdefault('ssm_expand_ratio', 2)
        model_config.setdefault('use_dct_hpm', True)
        model_config.setdefault('use_massm', True)
        model_config.setdefault('use_ssm_cross_gating', True)
        model_config.setdefault('use_apdm', True)
        model_config.update(kwargs)
    else:
        model_config = kwargs

    # ★ 应用消融实验配置 (覆盖对应开关)
    if config is not None:
        model_config['config'] = config

    return DFSSM(**model_config)
