"""
★ P9: 论文写作准备 - 综合评估脚本

功能:
1. 多指标评估: PSNR, SSIM, LPIPS (如可用), NIQE (无参考指标)
2. 消融实验: 通过配置开关控制每个模块, 一键运行7组实验
3. 结果可视化: 选取测试集中代表性图像, 对比展示输入/GT/各方法输出

用法:
    # 单模型评估
    python evaluate.py --checkpoint ./checkpoints/exp/best_model.pth --test_dir ./datasets/lol_v2/test

    # 消融实验 (自动运行7组配置)
    python evaluate.py --ablation --test_dir ./datasets/lol_v2/test --save_dir ./ablation_results

    # 结果可视化
    python evaluate.py --checkpoint ./checkpoints/exp/best_model.pth --test_dir ./datasets/lol_v2/test --visualize --num_vis 5
"""

import os
import argparse
import json
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import create_dfssm
from datasets import ImageRestorationDataset
from utils import calculate_psnr, calculate_ssim, inference, count_parameters


# ==================== LPIPS (可选) ====================
_lpips_net = None

def get_lpips_model(device):
    """懒加载 LPIPS 模型"""
    global _lpips_net
    if _lpips_net is not None:
        return _lpips_net
    try:
        import lpips
        _lpips_net = lpips.LPIPS(net='vgg').to(device)
        _lpips_net.eval()
        print("  ★ LPIPS 模型已加载 (VGG backbone)")
        return _lpips_net
    except ImportError:
        print("  ⚠️ lpips 库未安装, 跳过 LPIPS 评估 (pip install lpips)")
        return None
    except Exception as e:
        print(f"  ⚠️ LPIPS 加载失败: {e}")
        return None


def calculate_lpips(img1, img2, lpips_model, device):
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity)
    输入: [0, 1] 范围的 (B, C, H, W) tensor
    返回: float (越低越好, 0=完全相同)
    """
    if lpips_model is None:
        return float('nan')
    # LPIPS 期望 [-1, 1] 范围
    img1_norm = (img1 * 2 - 1).clamp(-1, 1)
    img2_norm = (img2 * 2 - 1).clamp(-1, 1)
    with torch.no_grad():
        dist = lpips_model(img1_norm, img2_norm)
    return dist.mean().item()


# ==================== NIQE (无参考指标, 简化版) ====================
def calculate_niqe(img, device):
    """
    简化版 NIQE (Natural Image Quality Evaluator) 估计
    使用图像统计特征 (均值、方差、偏度、峰度) 近似自然度
    输入: [0, 1] 范围的 (B, C, H, W) tensor
    返回: float (越低越好)
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.clamp(0, 1)
    # 转为灰度
    gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]

    # 计算局部统计特征 (近似 MSCN 系数)
    kernel_size = 7
    padding = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size ** 2)

    mu = F.conv2d(gray.unsqueeze(1), weight, padding=padding)
    mu_sq = mu ** 2
    sigma = torch.sqrt(F.conv2d(gray.unsqueeze(1) ** 2, weight, padding=padding) - mu_sq + 1e-8)
    mscn = (gray.unsqueeze(1) - mu) / (sigma + 1e-8)

    # 统计特征
    mscn_flat = mscn.clamp(-5, 5).flatten()
    mean_val = mscn_flat.mean().item()
    var_val = mscn_flat.var().item()
    # 偏度和峰度
    if var_val > 1e-8:
        skew = ((mscn_flat - mean_val) ** 3).mean().item() / (var_val ** 1.5)
        kurt = ((mscn_flat - mean_val) ** 4).mean().item() / (var_val ** 2) - 3.0
    else:
        return 10.0  # 退化图像, 返回较高值

    # 简化的 NIQE 估计 (越接近自然图像统计特征, 值越低)
    niqe_approx = abs(var_val - 1.0) * 3 + abs(skew) * 2 + abs(kurt) * 0.5
    return float(niqe_approx)


# ==================== 评估函数 ====================
@torch.no_grad()
def evaluate_model(model, test_loader, device, use_lpips=True, use_tile_inference=True, tile_size=512):
    """
    完整评估: PSNR + SSIM + LPIPS + NIQE
    """
    model.eval()
    lpips_model = get_lpips_model(device) if use_lpips else None

    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_list = []

    for input_img, target_img in tqdm(test_loader, desc='Evaluating', leave=False):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # 推理 (FP32, 与训练 --no_amp 一致)
        if use_tile_inference:
            output = inference(model, input_img, device=device, tile_size=tile_size, tile_overlap=192, use_amp=False)
        else:
            outputs_all = model(input_img)
            output = outputs_all[-1][0].clamp(0, 1)

        output = output.clamp(0, 1)
        target_img = target_img.clamp(0, 1)

        # 计算指标
        psnr = calculate_psnr(output, target_img)
        ssim = calculate_ssim(output, target_img)
        niqe = calculate_niqe(output, device)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        niqe_list.append(niqe)

        if lpips_model is not None:
            lp = calculate_lpips(output, target_img, lpips_model, device)
            lpips_list.append(lp)

    results = {
        'PSNR': float(np.mean(psnr_list)),
        'SSIM': float(np.mean(ssim_list)),
        'NIQE': float(np.mean(niqe_list)),
        'PSNR_std': float(np.std(psnr_list)),
        'SSIM_std': float(np.std(ssim_list)),
        'num_images': len(psnr_list),
    }
    if lpips_list:
        results['LPIPS'] = float(np.mean(lpips_list))
        results['LPIPS_std'] = float(np.std(lpips_list))

    return results


# ==================== TTA (Test-Time Augmentation) ====================
def _get_tta_transforms():
    """
    ★ 8-way D4 测试时增强 (dihedral group)
    返回: [(forward_fn, inverse_fn, name), ...]
    forward_fn(x): 对输入施加变换
    inverse_fn(x): 对模型输出施加逆变换, 还原到原图方向
    """
    def identity(x):
        return x
    def hflip(x):
        return torch.flip(x, dims=[-1])
    def vflip(x):
        return torch.flip(x, dims=[-2])
    def rot90(x):
        return torch.rot90(x, k=1, dims=[-2, -1])
    def rot180(x):
        return torch.rot90(x, k=2, dims=[-2, -1])
    def rot270(x):
        return torch.rot90(x, k=3, dims=[-2, -1])
    # hflip + rot90 的逆 = rot90 的逆 + hflip 的逆 = rot270 + hflip
    def hflip_rot90(x):
        return torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1])
    # hflip + rot270 的逆 = rot90 + hflip
    def hflip_rot270(x):
        return torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=[-2, -1])

    # (forward, inverse, name)
    # 逆变换: forward = A∘B (先 B 后 A), 则 inverse = B⁻¹∘A⁻¹
    #   forward_7 = hflip_rot90  = R90∘H   → inverse = H∘R270
    #   forward_8 = hflip_rot270 = R270∘H  → inverse = H∘R90
    return [
        (identity,      identity,        'orig'),
        (hflip,         hflip,           'hflip'),       # H⁻¹ = H
        (vflip,         vflip,           'vflip'),       # V⁻¹ = V
        (rot180,        rot180,          'rot180'),      # R180⁻¹ = R180
        (rot90,         rot270,          'rot90'),       # R90⁻¹ = R270
        (rot270,        rot90,           'rot270'),      # R270⁻¹ = R90
        (hflip_rot90,   lambda x: torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-1]), 'hflip+rot90'),   # (R90∘H)⁻¹ = H∘R270
        (hflip_rot270,  lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]), 'hflip+rot270'),  # (R270∘H)⁻¹ = H∘R90
    ]


@torch.no_grad()
def tta_inference(model, input_img, device, use_amp=False):
    """
    ★ 8-way TTA 推理
    对输入施加 8 种变换, 分别推理, 逆变换还原, 取平均
    支持 batch 输入

    Args:
        model: 模型
        input_img: (B, 3, H, W), 值域 [0, 1]
        device: 计算设备
    Returns:
        output: (B, 3, H, W), 值域 [0, 1], 8 种增强的平均
    """
    transforms = _get_tta_transforms()
    outputs_sum = None

    for forward_fn, inverse_fn, name in transforms:
        x_aug = forward_fn(input_img)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs_all = model(x_aug)
        out_aug = outputs_all[-1][0].float().clamp(0, 1)
        # 逆变换还原
        out_restored = inverse_fn(out_aug)
        if outputs_sum is None:
            outputs_sum = out_restored
        else:
            outputs_sum = outputs_sum + out_restored

    # 取平均
    output = (outputs_sum / len(transforms)).clamp(0, 1)
    return output


@torch.no_grad()
def evaluate_model_with_tta(model, test_loader, device, use_lpips=True,
                             use_tta=True, use_tile_inference=True, tile_size=512):
    """
    ★ 完整评估: 同时计算 无TTA 和 有TTA 的指标
    Returns:
        dict: {'no_tta': {...}, 'with_tta': {...}}
    """
    model.eval()
    lpips_model = get_lpips_model(device) if use_lpips else None

    no_tta_psnr, no_tta_ssim = [], []
    tta_psnr, tta_ssim = [], []
    per_image_results = []  # 保存每张图的指标 (供可视化用)

    for idx, (input_img, target_img) in enumerate(tqdm(test_loader, desc='Evaluating', leave=False)):
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        target_img = target_img.clamp(0, 1)

        # ★ 回退版 CGB 的 block_images 要求 H,W 被 grid 整除 (eval15 为 400x600,
        #   整图推理会报 einops Shape mismatch), 反射填充到 64 的倍数, 出图后裁回
        _, _, h_orig, w_orig = input_img.shape
        pad_h = (64 - h_orig % 64) % 64
        pad_w = (64 - w_orig % 64) % 64
        if pad_h or pad_w:
            input_img = torch.nn.functional.pad(
                input_img, (0, pad_w, 0, pad_h), mode='reflect')

        # === 无 TTA 推理 (FP32, 与训练 --no_amp 一致) ===
        if use_tile_inference:
            output_no_tta = inference(model, input_img, device=device,
                                       tile_size=tile_size, tile_overlap=192, use_amp=False)
        else:
            outputs_all = model(input_img)
            output_no_tta = outputs_all[-1][0].float()
        output_no_tta = output_no_tta.clamp(0, 1)
        if pad_h or pad_w:
            output_no_tta = output_no_tta[..., :h_orig, :w_orig]

        psnr_no = calculate_psnr(output_no_tta, target_img)
        ssim_no = calculate_ssim(output_no_tta, target_img)
        no_tta_psnr.append(psnr_no)
        no_tta_ssim.append(ssim_no)

        # === 有 TTA 推理 (FP32) ===
        if use_tta:
            output_tta = tta_inference(model, input_img, device, use_amp=False)
            if pad_h or pad_w:
                output_tta = output_tta[..., :h_orig, :w_orig]
            psnr_t = calculate_psnr(output_tta, target_img)
            ssim_t = calculate_ssim(output_tta, target_img)
            tta_psnr.append(psnr_t)
            tta_ssim.append(ssim_t)
        else:
            psnr_t, ssim_t = psnr_no, ssim_no

        per_image_results.append({
            'idx': idx,
            'psnr_no_tta': psnr_no,
            'psnr_tta': psnr_t,
            'ssim_no_tta': ssim_no,
            'ssim_tta': ssim_t,
            'input': input_img[0].cpu(),
            'output_no_tta': output_no_tta[0].cpu(),
            'output_tta': output_tta[0].cpu() if use_tta else output_no_tta[0].cpu(),
            'target': target_img[0].cpu(),
        })

    results = {
        'no_tta': {
            'PSNR': float(np.mean(no_tta_psnr)),
            'SSIM': float(np.mean(no_tta_ssim)),
            'PSNR_std': float(np.std(no_tta_psnr)),
            'SSIM_std': float(np.std(no_tta_ssim)),
            'num_images': len(no_tta_psnr),
        },
        'with_tta': {
            'PSNR': float(np.mean(tta_psnr)),
            'SSIM': float(np.mean(tta_ssim)),
            'PSNR_std': float(np.std(tta_psnr)),
            'SSIM_std': float(np.std(tta_ssim)),
            'num_images': len(tta_psnr),
        } if use_tta else None,
        'per_image': per_image_results,
    }

    # LPIPS (可选)
    if lpips_model is not None:
        lp_no, lp_t = [], []
        for r in per_image_results:
            o_no = r['output_no_tta'].unsqueeze(0).to(device)
            o_t = r['output_tta'].unsqueeze(0).to(device)
            t = r['target'].unsqueeze(0).to(device)
            lp_no.append(calculate_lpips(o_no, t, lpips_model, device))
            if use_tta:
                lp_t.append(calculate_lpips(o_t, t, lpips_model, device))
        results['no_tta']['LPIPS'] = float(np.mean(lp_no))
        if use_tta:
            results['with_tta']['LPIPS'] = float(np.mean(lp_t))

    return results


# ==================== 消融实验配置 ====================
ABLATION_CONFIGS = {
    'exp1_baseline': {
        'description': 'Baseline (所有新模块关闭)',
        'use_dct_hpm': False, 'use_massm': False, 'use_ssm_cross_gating': False, 'use_apdm': False,
    },
    'exp2_massm': {
        'description': '+MASSM',
        'use_dct_hpm': False, 'use_massm': True, 'use_ssm_cross_gating': False, 'use_apdm': False,
    },
    'exp3_massm_cgb': {
        'description': '+MASSM +Cross-Gating',
        'use_dct_hpm': False, 'use_massm': True, 'use_ssm_cross_gating': True, 'use_apdm': False,
    },
    'exp4_massm_cgb_apdm': {
        'description': '+MASSM +Cross-Gating +APDM',
        'use_dct_hpm': False, 'use_massm': True, 'use_ssm_cross_gating': True, 'use_apdm': True,
    },
    'exp5_full': {
        'description': '完整模型 (所有模块)',
        'use_dct_hpm': True, 'use_massm': True, 'use_ssm_cross_gating': True, 'use_apdm': True,
    },
    'exp6_full_minus_dct': {
        'description': '完整模型 - DCT-HPM',
        'use_dct_hpm': False, 'use_massm': True, 'use_ssm_cross_gating': True, 'use_apdm': True,
    },
    'exp7_full_minus_apdm': {
        'description': '完整模型 - APDM',
        'use_dct_hpm': True, 'use_massm': True, 'use_ssm_cross_gating': True, 'use_apdm': False,
    },
}


def run_ablation(args):
    """运行消融实验 (生成配置表, 供训练后评估)"""
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"★ 消融实验配置表 (共 {len(ABLATION_CONFIGS)} 组)")
    print(f"{'='*70}")

    ablation_table = []
    for name, config in ABLATION_CONFIGS.items():
        row = {
            'experiment': name,
            'description': config['description'],
            **{k: v for k, v in config.items() if k != 'description'},
        }
        ablation_table.append(row)

        # 生成对应的训练命令
        cmd = f"python train.py --train_dir {args.train_dir} --variant {args.variant} "
        cmd += f"--use_dct_hpm {int(config['use_dct_hpm'])} "
        cmd += f"--use_massm {int(config['use_massm'])} "
        cmd += f"--use_ssm_cross_gating {int(config['use_ssm_cross_gating'])} "
        cmd += f"--use_apdm {int(config['use_apdm'])} "
        cmd += f"--save_dir {args.save_dir}/{name} "
        cmd += f"--epochs {args.epochs} --lr {args.lr}"

        print(f"\n  [{name}] {config['description']}")
        print(f"    DCT-HPM={config['use_dct_hpm']}, MASSM={config['use_massm']}, "
              f"SSM-CGB={config['use_ssm_cross_gating']}, APDM={config['use_apdm']}")
        print(f"    CMD: {cmd}")

    # 保存配置表
    table_path = os.path.join(args.save_dir, 'ablation_configs.json')
    with open(table_path, 'w') as f:
        json.dump(ablation_table, f, indent=2)
    print(f"\n  ★ 配置表已保存: {table_path}")
    print(f"  每组实验训练完成后, 用以下命令评估:")
    print(f"    python evaluate.py --checkpoint <ckpt> --test_dir <test_dir> --save_dir {args.save_dir}/{name}")


# ==================== 结果可视化 ====================
@torch.no_grad()
def visualize_results(model, test_loader, device, save_dir, num_vis=5,
                       per_image_results=None):
    """
    ★ 4列对比可视化: 低光输入 | 输出(无TTA) | 输出(有TTA) | GT
    每张图标注 PSNR 值
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 如果有 per_image_results, 选 PSNR 提升最大的 num_vis 张图
    if per_image_results is not None and len(per_image_results) > 0:
        # 按 TTA 提升量排序 (提升大的优先展示)
        sorted_results = sorted(per_image_results,
                                 key=lambda r: r['psnr_tta'] - r['psnr_no_tta'],
                                 reverse=True)
        # 也混入几张 TTA 提升不大的 (随机选)
        vis_samples = sorted_results[:num_vis]
    else:
        vis_samples = None

    vis_count = 0
    if vis_samples is not None:
        for r in vis_samples:
            if vis_count >= num_vis:
                break
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            titles = [
                f"Input (Low-Light)",
                f"Output (No TTA)\nPSNR: {r['psnr_no_tta']:.2f} dB",
                f"Output (TTA)\nPSNR: {r['psnr_tta']:.2f} dB",
                f"GT (Normal-Light)",
            ]
            images = [r['input'], r['output_no_tta'], r['output_tta'], r['target']]

            for ax, img, title in zip(axes, images, titles):
                img_np = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ax.imshow(img_np)
                ax.set_title(title, fontsize=11)
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'vis_{vis_count+1:02d}_tta.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  ★ 可视化已保存: {save_path} "
                  f"(No TTA: {r['psnr_no_tta']:.2f} → TTA: {r['psnr_tta']:.2f} dB)")
            vis_count += 1
    else:
        for input_img, target_img in test_loader:
            if vis_count >= num_vis:
                break
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            output_no = inference(model, input_img, device=device, use_amp=True).clamp(0, 1)
            output_tta = tta_inference(model, input_img, device, use_amp=True).clamp(0, 1)
            target_img = target_img.clamp(0, 1)

            psnr_no = calculate_psnr(output_no, target_img)
            psnr_t = calculate_psnr(output_tta, target_img)

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            titles = [
                "Input (Low-Light)",
                f"Output (No TTA)\nPSNR: {psnr_no:.2f} dB",
                f"Output (TTA)\nPSNR: {psnr_t:.2f} dB",
                "GT (Normal-Light)",
            ]
            images = [input_img[0].cpu(), output_no[0].cpu(), output_tta[0].cpu(), target_img[0].cpu()]

            for ax, img, title in zip(axes, images, titles):
                img_np = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ax.imshow(img_np)
                ax.set_title(title, fontsize=11)
                ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'vis_{vis_count+1:02d}_tta.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  ★ 可视化已保存: {save_path}")
            vis_count += 1


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='★ P9: 论文评估与可视化')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型 checkpoint 路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--variant', type=str, default='M-2', help='模型变体')
    parser.add_argument('--save_dir', type=str, default='./eval_results', help='结果保存目录')
    parser.add_argument('--use_lpips', action='store_true', default=True, help='计算 LPIPS (需要安装 lpips 库)')
    parser.add_argument('--no_lpips', action='store_true', help='跳过 LPIPS')
    parser.add_argument('--visualize', action='store_true', help='生成结果可视化对比图')
    parser.add_argument('--num_vis', type=int, default=5, help='可视化图像数量')
    parser.add_argument('--ablation', action='store_true', help='生成消融实验配置表')
    parser.add_argument('--epochs', type=int, default=500, help='消融实验训练 epoch 数 (仅 --ablation 模式)')
    parser.add_argument('--lr', type=float, default=2e-4, help='消融实验学习率 (仅 --ablation 模式)')
    parser.add_argument('--batch_size', type=int, default=1, help='测试 batch size')
    parser.add_argument('--patch_size', type=int, default=256, help='测试 patch size (crop 模式使用)')
    parser.add_argument('--no_tta', action='store_true', help='禁用 TTA (默认启用 8-way TTA)')
    parser.add_argument('--full_image', action='store_true',
                        help='★ full-image 评估 (不裁剪, 返回整图, 与 SOTA 论文严格对齐)')
    # ★ Config-Driven: 消融评估 (与 train.py 开关一致)
    parser.add_argument('--use_dct_hpm', type=int, default=1, help='1=启用DCT-HPM, 0=关闭 (消融评估)')
    parser.add_argument('--use_massm', type=int, default=1, help='1=MASSM, 0=原MAB (消融评估)')
    parser.add_argument('--use_ssm_cross_gating', type=int, default=1, help='1=SSM-CGB, 0=原CGB (消融评估)')
    parser.add_argument('--use_apdm', type=int, default=1, help='1=启用APDM, 0=关闭 (消融评估)')
    parser.add_argument('--multi_gpu', action='store_true', help='使用 DataParallel 多卡评估')
    parser.add_argument('--gpu_ids', type=str, default=None, help='指定 GPU ID, 如 "0,1" (默认所有可见卡)')
    args = parser.parse_args()

    # ★ 多卡设置
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"  ★ CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 消融实验模式 ==========
    if args.ablation:
        run_ablation(args)
        return

    # ========== 单模型评估模式 ==========
    if args.checkpoint is None:
        print("错误: 评估模式需要 --checkpoint 参数")
        return

    use_tta = not args.no_tta

    print(f"\n{'='*60}")
    print(f"★ 模型评估 (TTA: {'ON' if use_tta else 'OFF'})")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test dir:   {args.test_dir}")
    print(f"  Device:     {device}")
    print(f"  TTA:        {'8-way D4' if use_tta else 'Disabled'}")
    if args.full_image:
        print(f"  Eval mode:  ★ FULL-IMAGE (不裁剪, 严格对标 SOTA)")
    else:
        print(f"  Eval mode:  Crop {args.patch_size}×{args.patch_size} (center-crop)")

    # 加载模型
    print("\n加载模型...")
    model = create_dfssm(
        variant=args.variant,
        config={
            'use_dct_hpm': bool(args.use_dct_hpm),
            'use_massm': bool(args.use_massm),
            'use_ssm_cross_gating': bool(args.use_ssm_cross_gating),
            'use_apdm': bool(args.use_apdm),
        },
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # ★ 预热: CrossGatingBlock/GetSpatialGatingWeights 的 norm/in_proj/out_proj
    #   等层是延迟初始化 (首次 forward 才建层), 不预热直接 load_state_dict
    #   会报 Unexpected keys
    with torch.no_grad():
        warm = max(256, int(args.patch_size))
        model(torch.randn(1, 3, warm, warm))
    model.load_state_dict(checkpoint['model_state_dict'])

    # ★ DataParallel 多卡包装
    if args.multi_gpu and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
        print(f"  ★ DataParallel: {n_gpus} GPUs")

    model = model.to(device)
    param_count = count_parameters(model)
    print(f"  参数量: {param_count/1e6:.2f}M")
    if 'best_psnr' in checkpoint:
        print(f"  Checkpoint best PSNR: {checkpoint['best_psnr']:.2f}")

    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = ImageRestorationDataset(
        root_dir=args.test_dir,
        patch_size=args.patch_size,
        is_train=False,
        augment=False,
        no_crop=args.full_image,  # ★ full-image: 不裁剪
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"  测试图像数: {len(test_dataset)}")

    # 评估
    print("\n开始评估...")
    use_lpips = args.use_lpips and not args.no_lpips

    if use_tta:
        # ★ 同时评估 无TTA 和 有TTA
        results = evaluate_model_with_tta(
            model, test_loader, device, use_lpips=use_lpips, use_tta=True
        )
        no_tta = results['no_tta']
        with_tta = results['with_tta']
        per_image = results.get('per_image', [])
    else:
        # 仅评估 无TTA
        results = evaluate_model(model, test_loader, device, use_lpips=use_lpips)
        no_tta = results
        with_tta = None
        per_image = []

    # 打印结果
    print(f"\n{'='*60}")
    print(f"★ 评估结果")
    print(f"{'='*60}")
    print(f"\n  [无 TTA]")
    print(f"    PSNR:  {no_tta['PSNR']:.2f} ± {no_tta['PSNR_std']:.2f} dB")
    print(f"    SSIM:  {no_tta['SSIM']:.4f} ± {no_tta['SSIM_std']:.4f}")
    if 'LPIPS' in no_tta:
        print(f"    LPIPS: {no_tta['LPIPS']:.4f} (越低越好)")
    print(f"    图像数: {no_tta['num_images']}")

    if with_tta is not None:
        print(f"\n  [有 TTA (8-way D4)]")
        print(f"    PSNR:  {with_tta['PSNR']:.2f} ± {with_tta['PSNR_std']:.2f} dB")
        print(f"    SSIM:  {with_tta['SSIM']:.4f} ± {with_tta['SSIM_std']:.4f}")
        if 'LPIPS' in with_tta:
            print(f"    LPIPS: {with_tta['LPIPS']:.4f} (越低越好)")
        print(f"    图像数: {with_tta['num_images']}")

        delta_psnr = with_tta['PSNR'] - no_tta['PSNR']
        delta_ssim = with_tta['SSIM'] - no_tta['SSIM']
        print(f"\n  ★ TTA 提升:")
        print(f"    ΔPSNR: {delta_psnr:+.4f} dB")
        print(f"    ΔSSIM: {delta_ssim:+.4f}")

    # 保存结果到 JSON
    results_path = os.path.join(args.save_dir, 'eval_results.json')
    save_data = {'no_tta': no_tta, 'with_tta': with_tta}
    # 移除 per_image 中的 tensor (无法序列化)
    save_data['per_image_summary'] = [
        {k: v for k, v in r.items() if k in ('idx', 'psnr_no_tta', 'psnr_tta',
                                              'ssim_no_tta', 'ssim_tta')}
        for r in per_image
    ]
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  ★ JSON 结果已保存: {results_path}")

    # ★ 保存结果到 test_results.txt
    txt_path = os.path.join(args.save_dir, 'test_results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("低光图像增强模型 - 测试集评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test dir:   {args.test_dir}\n")
        f.write(f"Variant:    {args.variant}\n")
        f.write(f"参数量:     {param_count/1e6:.2f}M\n")
        if 'best_psnr' in checkpoint:
            f.write(f"训练最佳PSNR: {checkpoint['best_psnr']:.2f} dB\n")
        f.write(f"测试图像数: {no_tta['num_images']}\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("[无 TTA]\n")
        f.write(f"  PSNR:  {no_tta['PSNR']:.2f} ± {no_tta['PSNR_std']:.2f} dB\n")
        f.write(f"  SSIM:  {no_tta['SSIM']:.4f} ± {no_tta['SSIM_std']:.4f}\n")
        if 'LPIPS' in no_tta:
            f.write(f"  LPIPS: {no_tta['LPIPS']:.4f}\n")
        if with_tta is not None:
            f.write("\n[有 TTA (8-way D4)]\n")
            f.write(f"  PSNR:  {with_tta['PSNR']:.2f} ± {with_tta['PSNR_std']:.2f} dB\n")
            f.write(f"  SSIM:  {with_tta['SSIM']:.4f} ± {with_tta['SSIM_std']:.4f}\n")
            if 'LPIPS' in with_tta:
                f.write(f"  LPIPS: {with_tta['LPIPS']:.4f}\n")
            f.write("\n[TTA 提升]\n")
            f.write(f"  ΔPSNR: {delta_psnr:+.4f} dB\n")
            f.write(f"  ΔSSIM: {delta_ssim:+.4f}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("逐图结果:\n")
        for r in per_image:
            f.write(f"  img {r['idx']:3d}: NoTTA PSNR={r['psnr_no_tta']:.2f} "
                    f"SSIM={r['ssim_no_tta']:.4f} | TTA PSNR={r['psnr_tta']:.2f} "
                    f"SSIM={r['ssim_tta']:.4f}\n")
        f.write("=" * 60 + "\n")
    print(f"  ★ 文本结果已保存: {txt_path}")

    # 可视化
    if args.visualize:
        print("\n生成结果可视化 (4列对比: 输入|无TTA|有TTA|GT)...")
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        visualize_results(model, test_loader, device, vis_dir,
                          num_vis=args.num_vis, per_image_results=per_image)

    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 延迟导入 matplotlib (仅在可视化时需要)
    if '--visualize' in __import__('sys').argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    main()
