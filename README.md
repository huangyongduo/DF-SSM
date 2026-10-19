# DF-SSM

PyTorch implementation of **Dual-Frequency State Space Model for Low-Light Enhancement via Topology-Preserving Spatial-Spectral Decoupling**.

DF-SSM achieves frequency-decoupled state transitions with linear computational complexity. It is built on a multi-stage, multi-scale U-Net backbone in which four modules are introduced:

- **DCT-HPM (DCT High-Pass Module)** — injects a learnable high-frequency prior before the first downsampling, retaining fine-grained textures against early attenuation.
- **MASSM (Multi-Axis State Space Mixer)** — traverses the state space simultaneously along two orthogonal axes, keeping global receptive fields without breaking the 2D neighborhood topology.
- **SSM-CGB (SSM Cross-Gating Block)** — supplies degradation-aware cross-scale fusion.
- **APDM (Amplitude–Phase Decoupling Module)** — decouples amplitude and phase, bounding phase residuals so that the resulting spatial attention maps suppress ringing.

## 项目结构

```
DF-SSM/
├── models/
│   ├── __init__.py       # 模型导出
│   ├── layers.py         # 基础层 (Conv, Norm, MLP, SE 等)
│   ├── blocks.py         # 基础模块 (MAB, CGB, SAM, Bottleneck)
│   ├── new_modules.py    # 四个核心新模块 (DCT-HPM, MASSM, SSM-CGB, APDM)
│   ├── dfssm.py          # DF-SSM 主网络 (多阶段多尺度 U-Net)
│   └── base_net.py       # 基础多轴网络 (消融基线, 不含四个新模块)
├── datasets/
│   └── __init__.py       # 数据集定义 (LOL-v1 / LOL-v2-Real / LOL-v2-Syn)
├── losses/
│   └── __init__.py       # 损失函数 (Charbonnier + SSIM + 频域一致性)
├── utils/
│   └── __init__.py       # 工具函数 (指标, 图像, 日志, 检查点)
├── train.py              # 训练脚本
├── evaluate.py           # 数据集评测脚本
├── inference_single.py   # 单图推理脚本
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 创建模型

```python
from models import create_dfssm

# 创建 DF-SSM (S-2 配置)
model = create_dfssm(variant='S-2', num_supervision_scales=3)

# 可用变体: S-1, S-2, S-3, M-1, M-2, M-3
# S = Small, M = Medium, 数字 = Stage 数量
```

消融开关（Config-Driven）：

```python
# 关闭某个模块（复现消融实验）
model = create_dfssm('S-2', config={
    'use_dct_hpm': True,
    'use_massm': True,
    'use_ssm_cross_gating': True,
    'use_apdm': False,
})
```

### 2. 前向传播

```python
import torch

# 输入: (B, 3, H, W)，H 和 W 需为 64 的倍数
x = torch.randn(1, 3, 256, 256)

outputs_all = model(x)
# outputs_all = [
#     [stage1_scale1, stage1_scale2, stage1_scale3],
#     [stage2_scale1, stage2_scale2, stage2_scale3],
# ]

final_output = outputs_all[-1][0]   # (B, 3, 256, 256)
```

### 3. 训练

```bash
python train.py \
    --train_dir datasets/LOLv2/Synthetic \
    --val_dir datasets/LOLv2/Synthetic/Test \
    --variant S-2 \
    --patch_size 384 \
    --batch_size 4 \
    --epochs 300 \
    --lr 2e-4 \
    --scheduler_type warmup_cosine \
    --grad_clip 1.0 \
    --use_dct_hpm 1 --use_massm 1 --use_ssm_cross_gating 1 --use_apdm 1
```

全图训练（消除训练尺度与全图推理的尺度不匹配）：

```bash
python train.py --full_image --train_dir ... --val_dir ... --variant S-2
```

数据集目录结构：

```
train/
├── low/      # 低光输入
└── high/     # 正常光目标
```

### 4. 评测 / 推理

```bash
# 数据集评测（--no_tta 关闭 8 路 D4 TTA，与论文指标对齐）
python evaluate.py \
    --checkpoint checkpoints/xxx/best_model.pth \
    --test_dir datasets/LOLv2/Synthetic/Test \
    --variant S-2 --full_image --no_tta

# 单图推理
python inference_single.py \
    --checkpoint checkpoints/xxx/best_model.pth \
    --input_image input.png \
    --output_dir ./results \
    --variant S-2
```

## 模型变体

| Variant | Features | Stages | Params (M) | FLOPs (G) |
|---------|----------|--------|------------|-----------|
| S-1     | 32       | 1      | ~6.1       | ~93       |
| S-2     | 32       | 2      | ~13.4      | ~207      |
| S-3     | 32       | 3      | ~20.6      | ~320      |
| M-1     | 64       | 1      | ~19.4      | ~308      |
| M-2     | 64       | 2      | ~40.8      | ~675      |
| M-3     | 64       | 3      | ~62.3      | ~1042     |

## 核心模块

### DCT-HPM
在第一次下采样之前，对输入做 DCT 高通分解并注入可学习高频先验，防止细节在早期下采样中被削弱。

### MASSM
以两条正交轴并行扫描状态空间（替换基础网络中的 MAB），在保持线性的同时获得全局感受野，且不破坏 2D 邻域拓扑。

### SSM-CGB
以状态空间门控替代交叉注意力，实现跨尺度、退化感知的条件化融合。

### APDM
在 Bottleneck 处解耦幅度与相位，约束相位残差，抑制空间注意力图在锐利边缘处产生的振铃伪影。

## 输入尺寸说明

输入图像 H、W 需为 64 的倍数（3 次下采样 + block/grid 划分）。推理时会自动 padding，输出时裁剪回原尺寸。

## Citation

```bibtex
@article{yi2026dfssm,
  title={Dual-Frequency State Space Model for Low-Light Enhancement via Topology-Preserving Spatial-Spectral Decoupling},
  author={Yi, Weichao and Huang, Yongduo and Zhang, Xiyi and Shi, Wenze and Fu, Yu and Yang, Yue and Lu, Chao},
  year={2026}
}
```
