"""
工具函数 - DF-SSM PyTorch实现
包含：图像保存、指标计算、日志、模型保存加载等
"""

import os
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ==================== 图像工具 ====================

def tensor_to_image(tensor):
    """
    将PyTorch tensor转为PIL Image
    tensor: (C, H, W) or (B, C, H, W), range [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一张
    
    tensor = tensor.clamp(0, 1)
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def save_image(tensor, path):
    """保存tensor为图像文件"""
    img = tensor_to_image(tensor)
    img.save(path)


def save_comparison(input_tensor, output_tensor, target_tensor, path):
    """保存对比图：输入 | 输出 | 目标"""
    input_img = tensor_to_image(input_tensor)
    output_img = tensor_to_image(output_tensor)
    target_img = tensor_to_image(target_tensor)
    
    w, h = input_img.size
    gap = 10
    total_w = w * 3 + gap * 2
    
    comparison = Image.new('RGB', (total_w, h), (255, 255, 255))
    comparison.paste(input_img, (0, 0))
    comparison.paste(output_img, (w + gap, 0))
    comparison.paste(target_img, (2 * (w + gap), 0))
    
    comparison.save(path)


# ==================== 评价指标 ====================

def _detect_range(tensor):
    """
    检测 tensor 的值域范围。
    如果 min < -0.5 或 max > 1.5，认为是 [-1, 1] 范围。
    """
    tmin = tensor.min().item()
    tmax = tensor.max().item()
    if tmin < -0.5 or tmax > 1.5:
        return 'minus_one_to_one'
    return 'zero_to_one'


def _to_zero_one(tensor):
    """将 tensor 映射到 [0, 1] 范围"""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def calculate_psnr(img1, img2, max_val=None):
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    
    自动检测输入值域:
    - 若输入在 [-1, 1] 范围，先映射到 [0, 1] 再计算
    - 若输入已在 [0, 1] 范围，直接计算
    
    Args:
        img1, img2: (B, C, H, W) 或 (C, H, W) tensors
        max_val: 像素最大值。None=自动检测(基于输入范围)
    Returns:
        psnr: float (dB)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    img1 = img1.float()
    img2 = img2.float()
    
    if max_val is None:
        r1 = _detect_range(img1)
        r2 = _detect_range(img2)
        if r1 == 'minus_one_to_one' or r2 == 'minus_one_to_one':
            img1 = _to_zero_one(img1)
            img2 = _to_zero_one(img2)
            max_val = 1.0
        else:
            max_val = 1.0
    
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-12)
    psnr = 20.0 * torch.log10(max_val / torch.sqrt(mse))
    psnr = torch.clamp(psnr, min=0.0, max=100.0)
    return torch.mean(psnr).item()


def calculate_ssim(img1, img2, window_size=11, max_val=None):
    """
    计算 SSIM (Structural Similarity Index)
    
    使用 2D 高斯加权窗口，标准 SSIM 公式。
    自动检测输入值域并映射到 [0, 1]。
    
    Args:
        img1, img2: (B, C, H, W) 或 (C, H, W) tensors
        window_size: 高斯窗口大小 (奇数)
        max_val: 像素最大值 (None=自动检测)
    Returns:
        ssim: float in [0, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    img1 = img1.float()
    img2 = img2.float()
    
    if max_val is None:
        r1 = _detect_range(img1)
        r2 = _detect_range(img2)
        if r1 == 'minus_one_to_one' or r2 == 'minus_one_to_one':
            img1 = _to_zero_one(img1)
            img2 = _to_zero_one(img2)
            max_val = 1.0
        else:
            max_val = 1.0
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_y = gauss.unsqueeze(1)
    window_x = gauss.unsqueeze(0)
    window_2d = window_y @ window_x
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window_2d = window_2d.expand(3, 1, window_size, window_size).contiguous()
    
    if window_2d.device != img1.device:
        window_2d = window_2d.to(img1.device)
    
    mu1 = F.conv2d(img1, window_2d, stride=1, padding=window_size//2, groups=3)
    mu2 = F.conv2d(img2, window_2d, stride=1, padding=window_size//2, groups=3)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window_2d, stride=1, padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window_2d, stride=1, padding=window_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window_2d, stride=1, padding=window_size//2, groups=3) - mu1_mu2
    
    sigma1_sq = torch.clamp(sigma1_sq, min=1e-8)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-8)
    sigma12 = torch.clamp(sigma12, min=-1e3, max=1e3)
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    denominator = torch.clamp(denominator, min=1e-8)
    
    ssim_map = numerator / denominator
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
    
    return torch.mean(ssim_map).item()


# ==================== 模型工具 ====================

def save_checkpoint(model, optimizer, scheduler, epoch, psnr, save_path, is_distributed=False):
    """保存训练检查点，支持DDP训练"""
    # 获取原始模型的state_dict（处理DDP和DataParallel包裹）
    if is_distributed:
        model_state = model.module.state_dict()
    elif hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_psnr': psnr,
    }
    torch.save(checkpoint, save_path)


def save_checkpoint_full(model, optimizer, scheduler, epoch,
                         best_psnr, best_ssim, best_val_loss, best_epoch,
                         save_path, is_distributed=False):
    """
    ★ 保存完整 checkpoint (包含所有最佳指标)
    用于 Early Stopping 的最佳模型保存
    """
    if is_distributed:
        model_state = model.module.state_dict()
    elif hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu', is_distributed=False):
    """加载训练检查点，支持DDP训练"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理DDP和DataParallel包裹的模型
    if is_distributed:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    elif hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0.0)
    
    return epoch, best_psnr


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 学习率调度 ====================

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    """
    带warmup的余弦学习率调度
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== 日志工具 ====================

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    """指标记录器"""
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics_dict, n=1):
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = AverageMeter()
            self.metrics[k].update(v, n)
    
    def get_avg(self):
        return {k: v.avg for k, v in self.metrics.items()}
    
    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()
    
    def __str__(self):
        items = [f"{k}: {v.avg:.4f}" for k, v in self.metrics.items()]
        return ", ".join(items)


# ==================== 推理工具 ====================

def pad_image(img, divisor=64):
    """
    将图像padding到可以被divisor整除的大小
    DF-SSM要求输入尺寸是64的倍数
    """
    b, c, h, w = img.shape
    
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    
    padding = (0, pad_w, 0, pad_h)
    # ★ AMP 兼容: reflect pad 不支持 FP16
    orig_dtype = img.dtype
    img = img.float()
    img_padded = F.pad(img, padding, mode='reflect')
    img_padded = img_padded.to(orig_dtype)
    
    return img_padded, (h, w)


def unpad_image(img, original_size):
    """移除padding"""
    h, w = original_size
    return img[:, :, :h, :w]


def create_gaussian_window(size, sigma=None):
    """创建高斯权重窗口，用于分块融合"""
    if sigma is None:
        sigma = size / 6.0
    
    coords = torch.arange(size).float() - size // 2
    gaussian = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    
    window_1d = gaussian.unsqueeze(0)
    window_2d = window_1d.t() @ window_1d
    
    return window_2d


@torch.no_grad()
def inference(model, img, device='cuda', divisor=64, tile_size=512, tile_overlap=192, use_amp=True):
    """
    单张图像推理（支持大图像分块推理）
    使用重叠分块+高斯融合消除马赛克伪影
    
    Args:
        model: DF-SSM模型
        img: 输入图像tensor (B, C, H, W)
        device: 运行设备
        divisor: padding除数
        tile_size: 分块大小
        tile_overlap: 重叠区域大小（推荐192）
        use_amp: 是否使用FP16混合精度推理
    """
    model.eval()
    
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    img = img.to(device)
    
    b, c, h, w = img.shape
    
    if h <= 4096 and w <= 4096:
        img_padded, original_size = pad_image(img, divisor)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs_all = model(img_padded)
        output = outputs_all[-1][0]
        output = unpad_image(output, original_size)
        return output.clamp(0, 1)
    
    img_padded, original_size = pad_image(img, divisor)
    _, _, h_padded, w_padded = img_padded.shape
    
    output = torch.zeros_like(img_padded)
    weights = torch.zeros_like(img_padded)
    
    gaussian_window = create_gaussian_window(tile_size, sigma=tile_size/4).to(device)
    gaussian_window = gaussian_window.unsqueeze(0).unsqueeze(0)
    
    stride = tile_size - tile_overlap
    
    for y in range(0, h_padded, stride):
        for x in range(0, w_padded, stride):
            y_end = min(y + tile_size, h_padded)
            x_end = min(x + tile_size, w_padded)
            
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = img_padded[:, :, y_start:y_end, x_start:x_end]
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                tile_outputs = model(tile)
            tile_output = tile_outputs[-1][0]
            
            win_h = y_end - y_start
            win_w = x_end - x_start
            win = gaussian_window[:, :, :win_h, :win_w]
            
            output[:, :, y_start:y_end, x_start:x_end] += tile_output * win
            weights[:, :, y_start:y_end, x_start:x_end] += win
    
    output = output / (weights + 1e-8)
    output = unpad_image(output, original_size)
    
    return output.clamp(0, 1)


def bilateral_filter(img, kernel_size=3, sigma_color=10.0, sigma_space=8.0):
    """
    轻量化双边滤波后处理
    仅平滑块边界噪点，不影响原图结构
    
    Args:
        img: 输入图像tensor (B, C, H, W) 或 (C, H, W)，值域[0, 1]
        kernel_size: 核大小
        sigma_color: 颜色空间标准差
        sigma_space: 空间距离标准差
    
    Returns:
        滤波后的图像tensor
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    b, c, h, w = img.shape
    img_np = img.cpu().numpy().transpose(0, 2, 3, 1)
    
    import cv2
    
    result = []
    for i in range(b):
        img_uint8 = (img_np[i] * 255).astype('uint8')
        filtered = cv2.bilateralFilter(img_uint8, kernel_size, sigma_color, sigma_space)
        result.append(torch.from_numpy(filtered.astype('float32') / 255.0).permute(2, 0, 1))
    
    return torch.stack(result).to(img.device)
