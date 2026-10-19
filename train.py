"""
训练脚本 - DF-SSM + APDM PyTorch实现
修复学习率崩溃、过拟合、验证震荡问题

修复要点：
1. T_max=500（总epoch数），确保LR单调递减，不反弹
2. weight_decay=0.1，更强正则化抑制过拟合
3. val_batch_size=4，增大验证batch减少震荡
4. 实现滑动平均平滑验证指标
5. eta_min=1e-6，防止LR完全归零

推荐配置：
- 初始学习率：4e-4（论文设置）
- Weight decay：0.1
- 学习率调度：CosineAnnealingLR，T_max=500
- Val batch size：4（减少验证震荡）
"""

import os
import argparse
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models import create_base_net
from datasets import ImageRestorationDataset, create_dataloader
from losses import MultiScaleLoss, CharbonnierLoss
from utils import (
    save_checkpoint, save_checkpoint_full, load_checkpoint, count_parameters,
    get_cosine_schedule_with_warmup, MetricsLogger,
    calculate_psnr, calculate_ssim
)


def parse_args():
    parser = argparse.ArgumentParser(description='DF-SSM Training')

    parser.add_argument('--train_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default=None, help='验证数据目录')
    # ★ P7: 多数据集联合训练
    parser.add_argument('--extra_train_dirs', type=str, default=None,
                        help='★ 额外训练数据集目录 (逗号分隔, 如 "path/to/sid_sony,path/to/exdark")')
    # ★ 数据集类型: auto(自动探测), lol, lol_v2, ve_lol
    parser.add_argument('--dataset_type', type=str, default='auto',
                        choices=['auto', 'lol', 'lol_v2', 've_lol'],
                        help='★ 数据集类型 (auto=自动探测目录结构)')
    parser.add_argument('--patch_size', type=int, default=256, help='训练patch大小(基础值,渐进训练会动态调整)')
    # ★ 全图训练: 不做任何裁剪, 直接以整图参与训练/验证
    #   用于消除"训练用小patch、推理用全图"的尺度不匹配(评测掉点)
    parser.add_argument('--full_image', action='store_true',
                        help='★ 全图训练/验证(不做任何裁剪). 让训练尺度与全图推理一致')
    # 关键修复：降低默认batch_size到2，防止OOM
    parser.add_argument('--batch_size', type=int, default=2, help='训练batch size(降低防止OOM)')
    parser.add_argument('--val_batch_size', type=int, default=4, help='验证batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数 (★ 多卡建议>=8, 配合 prefetch_factor=4)')

    parser.add_argument('--variant', type=str, default='M-2',
                        choices=['S-1', 'S-2', 'S-3', 'M-1', 'M-2', 'M-3'],
                        help='模型变体')
    parser.add_argument('--num_supervision_scales', type=int, default=3,
                        help='多尺度监督的尺度数量')

    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='★ 初始学习率 (频域模块降为2e-4, 原4e-4对SSM/FFT不稳)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='权重衰减 (AdamW)')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'warmup_cosine', 'cosine_restart', 'plateau', 'one_cycle'],
                        help='★ 学习率调度器: cosine/warmup_cosine/cosine_restart/plateau/one_cycle')
    parser.add_argument('--t_max', type=int, default=300, help='CosineAnnealingLR的T_max(必须=总epoch数)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup轮数(仅scheduler_type=cosine时无效)')
    parser.add_argument('--warmup_iters', type=int, default=500, help='★ 线性warmup迭代数 (500步≈5epochs, 原5000步太长)')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='最小学习率(防止LR完全归零)')
    # ★ CosineAnnealingWarmRestarts 参数
    parser.add_argument('--t_0', type=int, default=100, help='★ CosineWarmRestarts的T_0 (第一次重启周期)')
    parser.add_argument('--t_mult', type=int, default=2, help='★ CosineWarmRestarts的T_mult (重启周期乘数)')
    # ★ ReduceLROnPlateau 参数
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='★ Plateau LR衰减因子')
    parser.add_argument('--plateau_patience', type=int, default=20, help='★ Plateau耐心值(epoch)')
    # ★ OneCycleLR 参数 (推荐用于过拟合场景, 替代 cosine_restart)
    parser.add_argument('--pct_start', type=float, default=0.1, help='★ OneCycleLR warmup占比 (0.1=前10%%步数升到峰值)')
    parser.add_argument('--div_factor', type=float, default=10.0, help='★ OneCycleLR 初始LR=max_lr/div_factor')
    parser.add_argument('--final_div_factor', type=float, default=100.0, help='★ OneCycleLR 最终LR=max_lr/final_div_factor')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='★ 梯度裁剪max_norm(SSM梯度爆炸防线, 1.0=推荐)')
    parser.add_argument('--freq_lr_mult', type=float, default=0.5, help='★ 频域模块(APDM/DCT-HPM)学习率乘数 (0.5=半速)')

    # ★ 渐进式训练 (Progressive Training) - DF-SSM核心策略
    parser.add_argument('--progressive', action='store_true', help='★ 启用渐进式训练(小patch→大patch)')
    parser.add_argument('--prog_stage1_epochs', type=int, default=100, help='阶段1(128x128)训练epoch数')
    parser.add_argument('--prog_stage2_epochs', type=int, default=200, help='阶段2(256x256)训练epoch数')
    # 阶段3用 base patch_size (默认 512)

    # ★ 频域 Loss
    parser.add_argument('--freq_loss_weight', type=float, default=0.1, help='★ 频域Loss权重(强化APDM频域一致性)')
    parser.add_argument('--perceptual_weight', type=float, default=0.0, help='★ 感知Loss权重(VGG-19特征) (追PSNR/SSIM时建议0, 感知损失会牺牲PSNR换主观真实感)')
    parser.add_argument('--ssim_weight', type=float, default=0.05, help='★ SSIM Loss权重 (默认0.05; 追SSIM可调到 0.15~0.3, 但PSNR会略降)')

    # ★ Config-Driven 消融实验
    parser.add_argument('--use_dct_hpm', type=int, default=1, help='★ 1=启用DCT-HPM, 0=关闭')
    parser.add_argument('--use_massm', type=int, default=1, help='★ 1=MASSM, 0=原MAB')
    parser.add_argument('--use_ssm_cross_gating', type=int, default=1, help='★ 1=SSM-CGB, 0=原CGB')
    parser.add_argument('--use_apdm', type=int, default=1, help='★ 1=启用APDM, 0=关闭')
    parser.add_argument('--use_v2', type=int, default=1, help='★ 1=DFSSM, 0=基础网络')

    # ★ P6: 模型容量控制 (覆盖 variant 默认值, 用于简化模型防过拟合)
    parser.add_argument('--features', type=int, default=None,
                        help='★ 覆盖通道数 (如 32=减半, 64=默认, 128=加倍). None=用variant默认')
    parser.add_argument('--num_groups_override', type=int, default=None,
                        help='★ 覆盖每Stage的MASSM块数 (1=简化, 2=默认). None=用variant默认')
    parser.add_argument('--num_bottleneck_blocks_override', type=int, default=None,
                        help='★ 覆盖Bottleneck块数 (1=简化, 2=默认). None=用variant默认')

    # ★ 正则化: Dropout (防过拟合)
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='★ Dropout rate for MASSM/SSM-CGB (0=关闭, 0.1=推荐)')
    # ★ Label Smoothing (对 GT 施加微小扰动)
    # 注意: 对图像恢复/低光增强任务, 向 GT 加高斯噪声会污染干净监督目标并压低可达 PSNR,
    #       故默认关闭(0.0). 这是对 GT 的噪声注入而非分类意义上的 label smoothing.
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='★ GT 噪声扰动幅度 (默认0=关闭; 图像恢复任务不建议开启)')
    # ★ 正则化: Weight Decay 分组 (bias/norm 不衰减)
    parser.add_argument('--no_decay_bias_norm', action='store_true', default=True,
                        help='★ bias 和 LayerNorm/RMSNorm/BN 参数不做 weight decay')

    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--save_freq', type=int, default=20, help='保存频率(epoch)')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径(连同optimizer/scheduler/早停状态一起恢复,用于中断续训)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='★ 仅加载模型权重用于微调(不恢复optimizer/scheduler/best_*/start_epoch). 用于渐进式训练阶段2等场景. 自动处理 module. 前缀, strict=False')
    parser.add_argument('--pretrained_strict', action='store_true',
                        help='pretrained加载时是否强制严格匹配(strict=True). 默认False以容忍轻微结构差异')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001,
                        help='★ 早停最小提升幅度 (val_psnr提升<此值不算改善)')
    parser.add_argument('--val_interval', type=int, default=5, help='验证间隔(epoch数)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='自动划分验证集比例(当val_dir未提供时)')
    parser.add_argument('--smooth_window', type=int, default=5, help='滑动平均窗口大小')

    # 新增：梯度检查点（节省显存）
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='启用梯度检查点以节省显存（会降低训练速度约20%%）')
    parser.add_argument('--no_amp', action='store_true',
                        help='强制关闭混合精度训练 (解决FP16下溢导致的NaN问题)')

    # 多卡训练相关参数
    parser.add_argument('--multi_gpu', action='store_true', help='自动使用所有可用GPU')
    parser.add_argument('--gpu_ids', type=str, default=None, help='指定使用的GPU ID')
    # ★ 兼容 torchrun 三种传参方式:
    # 1. PyTorch >=1.10: 仅设置 LOCAL_RANK 环境变量 (我们之前的兼容)
    # 2. torch.distributed.run (PyTorch 2.1): 额外通过命令行传 --local-rank=N (连字符)
    #    若 argparse 不接受此写法, 会 exitcode=2: "unrecognized arguments: --local-rank=0"
    #    (argparse 会统一 dest=local_rank, 与下划线版本一致)
    # 3. 旧版 torch.distributed.launch: 传 --local_rank=N (下划线)
    _env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    parser.add_argument('--local_rank', '--local-rank', type=int, default=_env_local_rank,
                        dest='local_rank',
                        help='DDP自动传入的本地rank (torchrun通过LOCAL_RANK环境变量+--local-rank命令行双重传入)')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp(local_rank):
    """初始化分布式训练环境"""
    if local_rank >= 0:
        import torch.distributed as dist
        # 使用 torchrun 或 torch.distributed.launch 启动
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_ddp():
    """清理分布式训练环境"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass


def is_main_process():
    """判断是否为主进程（rank=0）"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except:
        pass
    return True


def get_rank():
    """获取当前进程的rank"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except:
        pass
    return 0


def get_world_size():
    """获取总进程数"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except:
        pass
    return 1


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch, args, step_per_epoch=True, grad_clip=0.5):
    model.train()
    metrics = MetricsLogger()

    grad_norm_history = {'freq': [], 'ssm': [], 'other': []}
    nan_batch_count = 0

    # ★ P8: 每10个epoch记录各模块输出范数 (诊断模块是否在学习)
    should_log_module_norms = (epoch % 10 == 0)
    module_norms_accum = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)

    # ★ 性能优化: 降低 GPU-CPU 同步频率, 避免 GPU 饿死
    # - loss 用 detach 累积 (tensor), 每 log_interval batch 才 .item() 一次
    # - psnr/ssim 每 log_interval batch 才算 (避免每 batch 同步)
    # - 移除周期性 empty_cache() (会让 CUDA 分配器阻塞)
    # - clip_grad_norm_ 已返回总梯度范数, 无需再遍历参数 .item()
    log_interval = 50
    loss_accum = 0.0       # tensor 累积 (detach), 避免每 batch .item() 同步
    loss_count = 0
    last_psnr = 0.0
    last_ssim = 0.0
    last_grad_norm = 0.0

    for batch_idx, (input_img, target_img) in enumerate(pbar):
        input_img = input_img.to(device, non_blocking=True)
        target_img = target_img.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=not args.no_amp):
            outputs_all = model(input_img)
            # ★ Label Smoothing: 对 GT 施加微小随机扰动 (隐式正则化, 防过拟合)
            if args.label_smoothing > 0:
                target_smoothed = target_img + torch.randn_like(target_img) * args.label_smoothing
                target_smoothed = torch.clamp(target_smoothed, 0.0, 1.0)
            else:
                target_smoothed = target_img
            loss, loss_dict = criterion(outputs_all, target_smoothed)

        if torch.isinf(loss) or torch.isnan(loss):
            nan_batch_count += 1
            if nan_batch_count <= 3:
                for i, stage_out in enumerate(outputs_all):
                    for j, scale_out in enumerate(stage_out):
                        if torch.isnan(scale_out).any():
                            print(f"  ⚠️ NaN in stage{i+1}_scale{j+1} output (shape={scale_out.shape})")
                        if torch.isinf(scale_out).any():
                            print(f"  ⚠️ Inf in stage{i+1}_scale{j+1} output (shape={scale_out.shape})")
                print(f"Warning: Inf/Nan loss at epoch {epoch}, batch {batch_idx}, skipping (×{nan_batch_count})")
            # ★ NaN 分支必须回收碎片: 坏张量/异常激活会碎片化显存, 后续batch容易OOM
            # (正常训练路径不建议 empty_cache, 但NaN分支是低频异常修复路径)
            del outputs_all, loss, loss_dict, input_img, target_img
            torch.cuda.empty_cache()
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # ★ clip_grad_norm_ 返回总梯度范数 (tensor), 替代原来遍历参数 .item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if step_per_epoch:
            scheduler.step()

        batch_size = input_img.size(0)
        # ★ loss 用 detach 累积 (不 .item(), 不阻塞 GPU)
        loss_accum = loss_accum + loss.detach()
        loss_count += 1
        metrics.update(loss_dict, batch_size)

        # ★ 每 log_interval batch 才算 psnr/ssim + 同步 loss + 更新进度条
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
            with torch.no_grad():
                final_output = outputs_all[-1][0]
                output_for_metric = torch.clamp(final_output, 0.0, 1.0)
                target_for_metric = torch.clamp(target_img, 0.0, 1.0)
                last_psnr = calculate_psnr(output_for_metric, target_for_metric)
                last_ssim = calculate_ssim(output_for_metric, target_for_metric)
            metrics.update({'psnr': last_psnr, 'ssim': last_ssim}, batch_size)

            avg_loss = (loss_accum / max(1, loss_count)).item()
            last_grad_norm = grad_norm.item()
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'psnr': f'{last_psnr:.2f}',
                'ssim': f'{last_ssim:.4f}',
                'lr': f'{current_lr:.6f}',
                'gn': f'{last_grad_norm:.2f}',
                'NaN': nan_batch_count
            })
            loss_accum = 0.0
            loss_count = 0

        del outputs_all, loss, loss_dict, input_img, target_img
    
    if nan_batch_count > 0:
        print(f"  ⚠️ Epoch {epoch}: {nan_batch_count}/{len(dataloader)} batches NaN/Inf loss")

    if any(grad_norm_history[k] for k in grad_norm_history):
        freq_avg = sum(grad_norm_history['freq']) / max(1, len(grad_norm_history['freq']))
        ssm_avg = sum(grad_norm_history['ssm']) / max(1, len(grad_norm_history['ssm']))
        other_avg = sum(grad_norm_history['other']) / max(1, len(grad_norm_history['other']))
        print(f"  GradNorm: freq={freq_avg:.3f}, ssm={ssm_avg:.3f}, other={other_avg:.3f}")

    # ★ P8: 每10个epoch输出各模块参数范数 (诊断模块是否在学习)
    if should_log_module_norms:
        module_norms = {}
        _raw_model = model.module if hasattr(model, 'module') else model
        for name, param in _raw_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 按模块类型分组
                module_type = 'other'
                name_lower = name.lower()
                if 'massm' in name_lower or 'mixer' in name_lower:
                    module_type = 'MASSM'
                elif 'cross_gating' in name_lower or 'ssm_cross' in name_lower:
                    module_type = 'SSM-CGB'
                elif 'apdm' in name_lower or 'freq' in name_lower:
                    module_type = 'APDM'
                elif 'dct' in name_lower:
                    module_type = 'DCT-HPM'

                if module_type not in module_norms:
                    module_norms[module_type] = []
                module_norms[module_type].append(param.data.norm().item())

        if module_norms:
            norm_str = []
            for mt in ['MASSM', 'SSM-CGB', 'APDM', 'DCT-HPM', 'other']:
                if mt in module_norms:
                    avg_norm = sum(module_norms[mt]) / len(module_norms[mt])
                    norm_str.append(f"{mt}={avg_norm:.3f}")
            print(f"  ★ ModuleNorms (epoch {epoch}): {', '.join(norm_str)}")

    torch.cuda.empty_cache()

    return metrics.get_avg()


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    metrics = MetricsLogger()
    nan_output_count = 0
    
    if dataloader is None or len(dataloader) == 0:
        return {'total': 0.0, 'psnr': 0.0, 'ssim': 0.0, '_empty': True}
    
    for batch_idx, (input_img, target_img) in enumerate(tqdm(dataloader, desc='Validation', leave=False)):
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs_all = model(input_img)
            loss, loss_dict = criterion(outputs_all, target_img)
        
        final_output = outputs_all[-1][0]
        
        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            nan_output_count += 1
            del outputs_all, loss, loss_dict, final_output, input_img, target_img
            continue
        
        output_for_metric = torch.clamp(final_output, 0.0, 1.0)
        target_for_metric = torch.clamp(target_img, 0.0, 1.0)
        psnr = calculate_psnr(output_for_metric, target_for_metric)
        ssim = calculate_ssim(output_for_metric, target_for_metric)
        
        if math.isnan(psnr) or math.isnan(ssim):
            nan_output_count += 1
            del outputs_all, loss, loss_dict, final_output, psnr, ssim, input_img, target_img
            continue
        
        batch_size = input_img.size(0)
        metrics.update(loss_dict, batch_size)
        metrics.update({'psnr': psnr, 'ssim': ssim}, batch_size)
        
        del outputs_all, loss, loss_dict, final_output, psnr, ssim
        del input_img, target_img
        
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    
    if nan_output_count > 0:
        print(f"  ⚠️ Validation: {nan_output_count} batches NaN/Inf output, skipped")
    
    result = metrics.get_avg()
    result['_empty'] = False
    return result


def main():
    # ========== 关键修复：设置CUDA内存分配器环境变量 ==========
    # ★ 只保留 max_split_size_mb:128, 不要和 expandable_segments 混用.
    #   PyTorch 2.1 (CUDA 12.1) 下, expandable_segments + max_split_size_mb 组合
    #   会触发 CUDACachingAllocator 内部断言: "!block->expandable_segment_"
    #   (在 SSM 模块的 torch.cat 分配大连续块时崩溃).
    #   单用 max_split_size_mb:128 已经足够解决碎片假 OOM.
    os.environ.setdefault(
        'PYTORCH_CUDA_ALLOC_CONF',
        'max_split_size_mb:128'
    )
    
    args = parse_args()
    
    # ========== 初始化分布式训练 ==========
    is_distributed = setup_ddp(args.local_rank)
    rank = get_rank()
    world_size = get_world_size()
    is_main = is_main_process()
    
    if is_main:
        print(f"Distributed training: {'Enabled' if is_distributed else 'Disabled'}")
        if is_distributed:
            print(f"  World size: {world_size}")
            print(f"  Rank: {rank}")
    
    set_seed(args.seed)
    
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(args.save_dir, f'dfssm_{args.variant}_{timestamp}')
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"Experiment directory: {exp_dir}")
        print(f"\nTraining configuration:")
        print(f"  - Patch size: {args.patch_size}")
        print(f"  - Batch size per GPU: {args.batch_size}")
        print(f"  - Total effective batch size: {args.batch_size * world_size}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Weight decay: {args.weight_decay}")
        print(f"  - Scheduler: {args.scheduler_type} (T_max={args.t_max if args.scheduler_type == 'cosine' else 'N/A'})")
        print(f"  - Warmup epochs: {args.warmup_epochs}")
        print(f"  - Total epochs: {args.epochs}")
        print(f"  - Early stop patience: {args.early_stop_patience}")
        print(f"  - Validation interval: every {args.val_interval} epochs")
    
    # ========== 设置设备 ==========
    if is_distributed:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        if args.multi_gpu and torch.cuda.device_count() > 1:
            gpu_ids = list(range(torch.cuda.device_count()))
            if args.gpu_ids:
                gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
            device = torch.device(f'cuda:{gpu_ids[0]}')
            if is_main:
                print(f"  - Using DataParallel with GPUs: {gpu_ids}")
        else:
            gpu_ids = [0]
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if is_main:
        print(f"  - Device: {device}")
    
    # ========== 创建数据集 ==========
    if is_main:
        print("\nCreating datasets...")
    
    # ★ 渐进式训练 (Progressive Training) - DF-SSM 核心策略
    # 逻辑: 前N个epoch用小patch(128), 中间M个epoch用中patch(256), 最后用大patch(512或原图)
    progressive_schedule = None
    if args.progressive and not args.full_image:
        progressive_schedule = [
            (0, 128),                                          # 阶段1: epoch 0~(stage1-1) 用 128x128
            (args.prog_stage1_epochs, 256),                    # 阶段2: epoch stage1~(stage1+stage2-1) 用 256x256
            (args.prog_stage1_epochs + args.prog_stage2_epochs, args.patch_size)  # 阶段3: 之后用基础patch_size(默认512)
        ]
        if is_main:
            print(f"  ★ 渐进式训练已启用: {progressive_schedule}")
    elif args.full_image:
        if is_main:
            print("  ★ 全图训练已启用, 渐进式训练被忽略")
    
    from datasets import ImageRestorationDataset, create_dataloader, split_train_val_dataset, MultiDatasetChain

    train_dataset = None
    val_dataset = None
    train_sampler = None
    val_sampler = None

    # ★ VE-LOL-L-Cap 结构: 同一根目录下同时含 train/test 子目录时,
    #   train_dir == val_dir 是正确用法(train 取 *_train, val 取 *_test), 不应触发 auto_split
    ve_lol_test_in = os.path.join(args.train_dir, 'VE-LOL-L-Cap-Low_test')
    ve_lol_test_gt = os.path.join(args.train_dir, 'VE-LOL-L-Cap-Normal_test')
    is_ve_lol_split = os.path.isdir(ve_lol_test_in) and os.path.isdir(ve_lol_test_gt)

    need_auto_split = (not args.val_dir) or (
        (args.val_dir == args.train_dir) and not is_ve_lol_split
    )

    if need_auto_split:
        if is_main:
            print(f"  ★ 未指定独立 val_dir，将从 train_dir 自动划分 {args.val_ratio*100:.0f}% 作为验证集")
        train_dataset, val_dataset = split_train_val_dataset(
            root_dir=args.train_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            patch_size=args.patch_size,
            enable_low_light_aug=False,
            no_crop=args.full_image,
            progressive_schedule=progressive_schedule
        )
    else:
        train_dataset = ImageRestorationDataset(
            root_dir=args.train_dir,
            patch_size=args.patch_size,
            is_train=True,
            augment=True,
            enable_low_light_aug=False,
            no_crop=args.full_image,
            progressive_schedule=progressive_schedule
        )
        val_dataset = ImageRestorationDataset(
            root_dir=args.val_dir,
            patch_size=args.patch_size,
            is_train=False,
            augment=False,
            no_crop=args.full_image
        )

    # ★ P7: 多数据集联合训练
    if args.extra_train_dirs:
        extra_dirs = [d.strip() for d in args.extra_train_dirs.split(',') if d.strip()]
        if extra_dirs:
            extra_datasets = [train_dataset]  # 第一个是主数据集
            for edir in extra_dirs:
                if is_main:
                    print(f"  ★ 加载额外训练数据集: {edir}")
                extra_ds = ImageRestorationDataset(
                    root_dir=edir,
                    patch_size=args.patch_size,
                    is_train=True,
                    augment=True,
                    enable_low_light_aug=False,
                    no_crop=args.full_image,
                    progressive_schedule=progressive_schedule
                )
                extra_datasets.append(extra_ds)
            # 用 MultiDatasetChain 串联, 按数据量比例采样
            train_dataset = MultiDatasetChain(extra_datasets, weights=None)
    
    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"Val samples: {len(val_dataset)}")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.val_batch_size,
                sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=4 if args.num_workers > 0 else None,
            )
        else:
            val_loader = None
    else:
        train_loader = create_dataloader(
            train_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=args.val_batch_size,
            num_workers=args.num_workers, shuffle=False, drop_last=False
        ) if val_dataset is not None else None
    
    # ========== 创建模型 ==========
    if is_main:
        print(f"\nCreating DF-SSM-{args.variant} model...")
    
    # ★ Config-Driven: 根据 --use_v2 决定使用 DF-SSM 还是基础网络
    if args.use_v2:
        # 使用 DF-SSM (集成 DCT-HPM, MASSM, SSM-CGB, APDM)
        from models import create_dfssm
        
        # 构建消融实验配置
        v2_config = {
            'use_dct_hpm': bool(args.use_dct_hpm),
            'use_massm': bool(args.use_massm),
            'use_ssm_cross_gating': bool(args.use_ssm_cross_gating),
            'use_apdm': bool(args.use_apdm),
        }
        if is_main:
            print(f"  ★ 使用 DF-SSM (Config-Driven)")
            print(f"    Config: {v2_config}")
        
        # ★ P6: 模型容量覆盖参数
        capacity_kwargs = {}
        if args.features is not None:
            capacity_kwargs['features'] = args.features
        if args.num_groups_override is not None:
            capacity_kwargs['num_groups'] = args.num_groups_override
        if args.num_bottleneck_blocks_override is not None:
            capacity_kwargs['num_bottleneck_blocks'] = args.num_bottleneck_blocks_override

        model = create_dfssm(
            variant=args.variant,
            num_supervision_scales=args.num_supervision_scales,
            config=v2_config,  # ★ 传入消融实验配置
            dropout_rate=args.dropout_rate,  # ★ 正则化: Dropout
            **capacity_kwargs,  # ★ P6: 模型容量控制
        )
        if is_main and capacity_kwargs:
            print(f"    ★ 容量覆盖: {capacity_kwargs}")
    else:
        # 使用基础网络
        if is_main:
            print(f"  使用基础网络")
        model = create_base_net(
            variant=args.variant,
            num_supervision_scales=args.num_supervision_scales
        )
    
    # ========== 关键修复：启用梯度检查点以节省显存 ==========
    # 梯度检查点会用计算换空间，降低约20%显存使用
    # 代价是训练速度降低约20%
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main:
            print("  [Gradient Checkpointing] Enabled to save GPU memory")
    
    with torch.no_grad():
        model(torch.randn(1, 3, args.patch_size, args.patch_size))
    
    model = model.to(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 使用DDP包裹模型（延迟导入）
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # ★ OOM 防护:
        # - find_unused_parameters=False: 省掉 unused param 扫描 + 临时桶显存(~500MB/卡)
        #   DF-SSM 所有参数都会参与反向传播, 不存在 unused param, 关掉安全
        # - gradient_as_bucket_view=True: grad bucket 直接复用 param grad 内存视图,
        #   省掉 DDP 内部临时梯度备份张量 (每个 bucket 省 C×B 大小, 约 300-800MB/卡)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    elif args.multi_gpu and torch.cuda.device_count() > 1 and not is_distributed:
        # DataParallel模式（仅作为备选）
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    num_params = count_parameters(model)
    if is_main:
        print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # ========== 损失函数 ==========
    # ★ 主 Loss 使用 Charbonnier Loss (L1 的平滑版本)
    # ★ 辅助 Loss: SSIM (--ssim_weight, 默认0.05) + 频域 Loss (--freq_loss_weight, 默认0.1)
    base_loss = CharbonnierLoss()
    criterion = MultiScaleLoss(
        base_loss=base_loss,
        l1_weight=0.0,
        ssim_weight=args.ssim_weight,
        psnr_weight=0.0,
        frequency_weight=args.freq_loss_weight,
        perceptual_weight=args.perceptual_weight
    )
    criterion = criterion.to(device)
    if is_main:
        print(f"  ★ 损失函数: Charbonnier + {args.ssim_weight}*SSIM + {args.freq_loss_weight}*FreqLoss + {args.perceptual_weight}*PerceptualLoss")
    
    # ========== 优化器和调度器 ==========
    # ★ 频域模块(APDM/DCT-HPM)使用独立学习率 (freq_lr_mult=0.5)
    # 原因: FFT/DCT 操作对学习率敏感, 过大 LR 会导致频域能量爆炸
    # ★ 正则化: bias 和 LayerNorm/RMSNorm/BN 参数不做 weight decay
    freq_lr = args.lr * args.freq_lr_mult
    freq_decay_params = []    # 频域模块 - 有 weight decay
    freq_no_decay_params = [] # 频域模块 - 无 weight decay (bias/norm)
    other_decay_params = []   # 主干模块 - 有 weight decay
    other_no_decay_params = [] # 主干模块 - 无 weight decay (bias/norm)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # ★ 判断是否为 bias 或 norm 参数 (不做 weight decay)
        is_no_decay = False
        if name.endswith('.bias'):
            is_no_decay = True
        elif any(nd in name for nd in ['norm', 'Norm', 'ln_', 'bn_', 'gn_', 'rmsnorm', 'RMSNorm']):
            is_no_decay = True

        # 识别频域模块参数: APDM 和 DCT-HPM
        # APDM/频域/DCT 模块参数 → 使用独立(0.5x)学习率. 靠 'apdm' 容器名匹配,
        # 否则 amp_conv/phase_conv/attn_conv 等参数无法被识别为频域参数, 频域降速失效.
        is_freq = ('freq' in name.lower() or 'dct' in name.lower()
                   or 'fft' in name.lower() or 'apdm' in name.lower())

        if is_freq:
            if is_no_decay:
                freq_no_decay_params.append(param)
            else:
                freq_decay_params.append(param)
        else:
            if is_no_decay:
                other_no_decay_params.append(param)
            else:
                other_decay_params.append(param)

    param_groups = [
        {'params': other_decay_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': other_no_decay_params, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': freq_decay_params, 'lr': freq_lr, 'weight_decay': args.weight_decay},
        {'params': freq_no_decay_params, 'lr': freq_lr, 'weight_decay': 0.0},
    ]
    
    if is_main:
        freq_count = sum(p.numel() for p in freq_decay_params + freq_no_decay_params)
        other_count = sum(p.numel() for p in other_decay_params + other_no_decay_params)
        no_decay_count = sum(p.numel() for p in freq_no_decay_params + other_no_decay_params)
        print(f"  ★ 优化器: 主干LR={args.lr:.2e}, 频域模块LR={freq_lr:.2e} (×{args.freq_lr_mult})")
        print(f"      频域参数量: {freq_count/1e3:.1f}K, 主干参数量: {other_count/1e6:.2f}M")
        print(f"      ★ Weight Decay 分组: decay={other_count-no_decay_count+freq_count-len(freq_no_decay_params):,}params, "
              f"no_decay(bias/norm)={no_decay_count:,}params")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # ★ 学习率调度器 (顶会级训练策略)
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.t_max,
            eta_min=args.eta_min
        )
        step_per_epoch = False
    elif args.scheduler_type == 'cosine_restart':
        # ★ CosineAnnealingWarmRestarts: 每 T_0 → T_0*T_mult → ... epoch 重启一次
        # 重启时 LR 跳回峰值, 帮助模型逃离局部最优, 缓解 epoch 200 后的过拟合
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t_0,
            T_mult=args.t_mult,
            eta_min=args.eta_min
        )
        step_per_epoch = False
        if is_main:
            restarts = []
            t_cur = args.t_0
            cumulative = 0
            while cumulative < args.epochs:
                restarts.append(f"epoch {cumulative+t_cur}")
                cumulative += t_cur
                t_cur *= args.t_mult
            print(f"  ★ CosineWarmRestarts: T_0={args.t_0}, T_mult={args.t_mult}, "
                  f"restarts at: {restarts[:5]}")
    elif args.scheduler_type == 'plateau':
        # ★ ReduceLROnPlateau: 监控 Val Loss, 连续 patience 个 epoch 无改善则衰减 LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.eta_min,
            verbose=True
        )
        step_per_epoch = False
        if is_main:
            print(f"  ★ ReduceLROnPlateau: factor={args.plateau_factor}, "
                  f"patience={args.plateau_patience}, min_lr={args.eta_min}")
    elif args.scheduler_type == 'one_cycle':
        # ★ OneCycleLR: 单次warmup→峰值→cosine退火, 适合过拟合场景
        # 不像 WarmRestarts 会反复冲击, OneCycleLR 只做一次平滑的升-降
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],  # 支持参数组(主干+频域)
            total_steps=total_steps,
            pct_start=args.pct_start,
            anneal_strategy='cos',
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
        )
        step_per_epoch = True  # ★ 每个 step 调用 scheduler.step()
        if is_main:
            init_lr = args.lr / args.div_factor
            final_lr = args.lr / args.final_div_factor
            print(f"  ★ OneCycleLR: max_lr={args.lr}, total_steps={total_steps}, "
                  f"pct_start={args.pct_start}")
            print(f"    LR轨迹: {init_lr:.2e} → {args.lr:.2e}(峰值) → {final_lr:.2e}(最终)")
            print(f"    warmup步数={int(total_steps*args.pct_start)} "
                  f"(≈{total_steps*args.pct_start/max(1,len(train_loader)):.1f} epochs)")
    else:
        # ★ Warmup + Cosine: 基于迭代步数
        total_steps = len(train_loader) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=total_steps
        )
        step_per_epoch = True
        if is_main:
            iters_per_epoch = len(train_loader)
            print(f"  ★ Warmup+Cosine: warmup={args.warmup_iters}步(≈{args.warmup_iters/max(1,iters_per_epoch):.1f}epochs), total={total_steps}步")
    
    # 兼容不同 PyTorch 版本的 GradScaler
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    
    # ========== 恢复训练 ==========
    start_epoch = 0
    best_psnr = 0.0       # ★ 最佳 Val PSNR (原始值, 用于保存最佳模型)
    best_ssim = 0.0       # ★ 最佳 Val SSIM
    best_val_loss = float('inf')  # ★ 最佳 Val Loss (辅助监控)
    best_epoch = 0        # ★ 最佳指标出现的 epoch
    early_stop_counter = 0
    
    # ========== 滑动平均缓存（用于平滑验证指标）==========
    # 使用deque存储最近的验证结果，实现滑动平均
    from collections import deque
    smooth_window = args.smooth_window  # 滑动窗口大小，默认5
    val_psnr_history = deque(maxlen=smooth_window)  # 存储最近N次的Val PSNR
    val_ssim_history = deque(maxlen=smooth_window)  # 存储最近N次的Val SSIM
    val_loss_history = deque(maxlen=smooth_window)  # 存储最近N次的Val Loss
    
    # ★ 仅加载模型权重用于微调 (不恢复 optimizer/scheduler/早停/start_epoch)
    # 用于渐进式训练阶段2: 加载阶段1权重, 但学习率/调度器/早停计数全部重置
    if args.pretrained:
        if is_main:
            print(f"Loading pretrained weights from {args.pretrained} (微调模式, 重置优化器/调度器/早停)...")
        ckpt = torch.load(args.pretrained, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        # 自动剥离 DataParallel/DDP 的 'module.' 前缀
        new_sd = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith('module.') else k
            new_sd[nk] = v
        # 取目标模型 (剥离外层 DataParallel/DDP)
        target = model.module if hasattr(model, 'module') else model
        missing, unexpected = target.load_state_dict(new_sd, strict=args.pretrained_strict)
        if is_main:
            print(f"  ✓ 已加载 {len(new_sd)} 个参数张量 (strict={args.pretrained_strict})")
            if missing:
                print(f"  ⚠ Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(f"  → start_epoch=0, best_psnr=0.0, best_ssim=0.0 (全部重置, 从头微调)")
        # 微调模式: 所有 best 指标和 start_epoch 保持初始值, 不从 checkpoint 恢复

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}...")
        start_epoch, best_psnr = load_checkpoint(
            model, args.resume, optimizer, scheduler, device, is_distributed
        )
        # ★ 从 checkpoint 恢复其余最佳指标
        try:
            ckpt = torch.load(args.resume, map_location='cpu')
            best_ssim = ckpt.get('best_ssim', 0.0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            best_epoch = ckpt.get('best_epoch', start_epoch)
        except Exception:
            pass
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}, best SSIM: {best_ssim:.4f}")
    
    # ========== 训练循环 ==========
    if is_main:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        
        # ★ 渐进式训练: 根据 epoch 动态调整 patch_size
        # 在每个 epoch 开始前调用 set_epoch, Dataset 内部会更新 self.patch_size
        if progressive_schedule is not None:
            # ★ 兼容 MultiDatasetChain (调用其 set_epoch 会传播到所有子数据集)
            if hasattr(train_dataset, 'set_epoch') and hasattr(train_dataset, 'datasets'):
                # MultiDatasetChain
                train_dataset.set_epoch(epoch)
                _ds = train_dataset.datasets[0]
                _ds = _ds.dataset if hasattr(_ds, 'dataset') else _ds
                new_ps = _ds.patch_size
                if is_main:
                    print(f"  ★ 渐进式训练: patch_size →{new_ps} (epoch {epoch}, MultiDatasetChain)")
            else:
                _ds = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
                old_ps = _ds.patch_size
                _ds.set_epoch(epoch)
                new_ps = _ds.patch_size
                if new_ps != old_ps and is_main:
                    print(f"  ★ 渐进式训练: patch_size {old_ps}→{new_ps} (epoch {epoch})")
        
        # DDP需要在每个epoch设置sampler的epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch,
            args=args,  # ★ 传入 args (用于 label_smoothing)
            step_per_epoch=step_per_epoch,
            grad_clip=args.grad_clip  # ★ 传入梯度裁剪阈值
        )
        
        if not step_per_epoch and args.scheduler_type != 'plateau':
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        if is_main:
            print(f"\nEpoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['total']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}, "
                  f"SSIM: {train_metrics.get('ssim', 0.0):.4f}")
        
        # ========== 验证 ==========
        should_validate = (val_loader is not None) and (epoch % args.val_interval == 0)
        if should_validate:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            val_metrics = validate(model, val_loader, criterion, device, use_amp=not args.no_amp)
            
            if val_metrics.get('_empty', False):
                if is_main:
                    print(f"  ⚠️ Validation dataloader returned empty, skipping val metrics")
                should_validate = False
            else:
                if is_distributed:
                    val_metrics = reduce_dict(val_metrics, world_size)
                
                val_psnr_history.append(val_metrics['psnr'])
                val_ssim_history.append(val_metrics['ssim'])
                val_loss_history.append(val_metrics['total'])
                
                smooth_val_psnr = sum(val_psnr_history) / len(val_psnr_history)
                smooth_val_ssim = sum(val_ssim_history) / len(val_ssim_history)
                smooth_val_loss = sum(val_loss_history) / len(val_loss_history)
                
                if is_main:
                    print(f"  Val   - Loss: {val_metrics['total']:.4f}(平滑:{smooth_val_loss:.4f}), "
                          f"PSNR: {val_metrics['psnr']:.2f}(平滑:{smooth_val_psnr:.2f}), "
                          f"SSIM: {val_metrics['ssim']:.4f}(平滑:{smooth_val_ssim:.4f})")

                    # ★ 早停逻辑: 仅监控 Val PSNR (主指标)
                    # min_delta: PSNR 提升幅度 < min_delta 不算改善
                    # patience: 连续 patience 次验证无改善则停止
                    raw_val_psnr = val_metrics['psnr']
                    raw_val_loss = val_metrics['total']
                    raw_val_ssim = val_metrics['ssim']

                    # ★ 仅当 Val PSNR 提升 > min_delta 才算改善
                    psnr_improvement = raw_val_psnr - best_psnr
                    psnr_improved = psnr_improvement > args.early_stop_min_delta

                    # 辅助指标仍记录最佳值 (但不触发早停重置)
                    if raw_val_ssim > best_ssim:
                        best_ssim = raw_val_ssim
                    if raw_val_loss < best_val_loss:
                        best_val_loss = raw_val_loss

                    if psnr_improved:
                        best_psnr = raw_val_psnr
                        best_epoch = epoch
                        best_path = os.path.join(exp_dir, 'best_model.pth')
                        # ★ 保存完整 checkpoint: 模型权重 + 优化器 + 调度器 + epoch + 最佳指标
                        save_checkpoint_full(
                            model, optimizer, scheduler, epoch,
                            best_psnr, best_ssim, best_val_loss, best_epoch,
                            best_path, is_distributed
                        )
                        print(f"  ★ New best model! Val PSNR: {best_psnr:.4f} "
                              f"(+{psnr_improvement:.4f} > min_delta={args.early_stop_min_delta}), "
                              f"SSIM: {best_ssim:.4f}, Loss: {best_val_loss:.4f} "
                              f"(epoch {best_epoch})")
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= args.early_stop_patience:
                            print(f"\n{'='*60}")
                            print(f"⛔ Early stopping at epoch {epoch}, best Val PSNR: {best_psnr:.2f}")
                            print(f"   No PSNR improvement (>{args.early_stop_min_delta}) "
                                  f"for {args.early_stop_patience} validation periods")
                            print(f"   Best Val PSNR:  {best_psnr:.4f} dB (epoch {best_epoch})")
                            print(f"   Best Val SSIM:  {best_ssim:.4f}")
                            print(f"   Best Val Loss:  {best_val_loss:.4f}")
                            print(f"{'='*60}")

                            final_path = os.path.join(exp_dir, 'final_model.pth')
                            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, final_path, is_distributed)
                            print(f"Training completed early at epoch {epoch}!")

                            # ★ 加载最佳 checkpoint 并输出最终测试结果
                            best_ckpt_path = os.path.join(exp_dir, 'best_model.pth')
                            if os.path.exists(best_ckpt_path):
                                print(f"\nLoading best checkpoint from {best_ckpt_path} for final evaluation...")
                                load_checkpoint(model, best_ckpt_path, device=device, is_distributed=is_distributed)
                                if is_main:
                                    print(f"  ★ Best model loaded (epoch {best_epoch}, PSNR={best_psnr:.2f})")

                            print(f"Final model saved to {final_path}")
                            cleanup_ddp()
                            return

            # ★ ReduceLROnPlateau: 在验证后调用 scheduler.step(val_loss)
            if args.scheduler_type == 'plateau' and not val_metrics.get('_empty', False):
                scheduler.step(val_metrics['total'])

        # ========== 写日志 ==========
        if is_main:
            log_path = os.path.join(exp_dir, 'training_log.csv')
            write_header = not os.path.exists(log_path)

            current_lr = optimizer.param_groups[0]['lr']

            with open(log_path, 'a') as f:
                if write_header:
                    f.write('epoch,train_loss,train_psnr,train_ssim,')
                    f.write('val_loss_raw,val_psnr_raw,val_ssim_raw,')
                    f.write('val_loss_smooth,val_psnr_smooth,val_ssim_smooth,')
                    f.write('lr\n')  # ★ P8: 新增学习率列

                train_ssim = train_metrics.get('ssim', 0.0)
                if should_validate:
                    f.write(f'{epoch},{train_metrics["total"]:.4f},'
                            f'{train_metrics["psnr"]:.2f},{train_ssim:.4f},'
                            f'{val_metrics["total"]:.4f},{val_metrics["psnr"]:.2f},{val_metrics["ssim"]:.4f},'
                            f'{smooth_val_loss:.4f},{smooth_val_psnr:.2f},{smooth_val_ssim:.4f},'
                            f'{current_lr:.8f}\n')
                else:
                    f.write(f'{epoch},{train_metrics["total"]:.4f},'
                            f'{train_metrics["psnr"]:.2f},{train_ssim:.4f},'
                            f',,,,,,,'
                            f'{current_lr:.8f}\n')
        
        # ========== 保存checkpoint ==========
        if is_main and epoch % args.save_freq == 0:
            save_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, save_path, is_distributed)
            print(f"  Checkpoint saved: {save_path}")
    
    # ========== 训练结束 ==========
    if is_main:
        final_path = os.path.join(exp_dir, 'final_model.pth')
        save_checkpoint(model, optimizer, scheduler, args.epochs, best_psnr, final_path, is_distributed)

        # ★ 加载最佳 checkpoint 用于最终评估
        best_ckpt_path = os.path.join(exp_dir, 'best_model.pth')
        if os.path.exists(best_ckpt_path):
            print(f"\nLoading best checkpoint from {best_ckpt_path} for final evaluation...")
            load_checkpoint(model, best_ckpt_path, device=device, is_distributed=is_distributed)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Final model saved to: {final_path}")
        print(f"Best Val PSNR:  {best_psnr:.2f} dB (epoch {best_epoch})")
        print(f"Best Val SSIM:  {best_ssim:.4f}")
        print(f"Best Val Loss:  {best_val_loss:.4f}")
        print(f"{'='*60}")
    
    cleanup_ddp()


def reduce_dict(input_dict, world_size):
    """DDP下聚合各进程的指标 (跳过布尔等非数值项, 避免 all_reduce 不支持 bool)"""
    import torch.distributed as dist
    output_dict = {}
    for k, v in input_dict.items():
        if not isinstance(v, (int, float)):
            output_dict[k] = v
            continue
        t = torch.tensor([v], device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        output_dict[k] = t.item() / world_size
    return output_dict


if __name__ == '__main__':
    main()