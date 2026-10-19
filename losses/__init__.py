"""
损失函数定义 - DF-SSM PyTorch实现
包含：Charbonnier Loss、SSIM Loss、Perceptual Loss、频域 Loss、多尺度损失等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 Loss的平滑版本)
    L(x) = sqrt(x^2 + epsilon^2)
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        loss = torch.clamp(loss, min=1e-6, max=10.0)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    SSIM Loss - 基于高斯加权窗口的结构相似性损失
    值域: 输入应为 [0, 1] 范围
    """
    def __init__(self, window_size=11, sigma=1.5, eps=1e-8):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.eps = eps
        self.window = self._create_window(window_size)
    
    def _create_window(self, window_size):
        # ★ 修复: 原实现只取 arange(window_size) 的正半段, 得到的是单边衰减窗,
        #   导致 SSIM 卷积偏向图像左上角, 度量失真. 标准 SSIM 需以中心对称的高斯窗.
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        gauss = gauss / gauss.sum()
        window_y = gauss.unsqueeze(1)
        window_x = gauss.unsqueeze(0)
        window_2d = window_y @ window_x
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)
        window_2d = window_2d.expand(3, 1, window_size, window_size).contiguous()
        return window_2d
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        target = torch.clamp(target, min=self.eps, max=1.0)
        
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        b, c, h, w = pred.shape
        
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self.window.to(device=pred.device, dtype=pred.dtype)
        
        mu_pred = F.conv2d(pred, self.window, padding=self.window_size//2, groups=c)
        mu_target = F.conv2d(target, self.window, padding=self.window_size//2, groups=c)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=c) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=c) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=c) - mu_pred_target
        
        sigma_pred_sq = torch.clamp(sigma_pred_sq, min=self.eps)
        sigma_target_sq = torch.clamp(sigma_target_sq, min=self.eps)
        sigma_pred_target = torch.clamp(sigma_pred_target, min=-1e3, max=1e3)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        denominator = torch.clamp(denominator, min=self.eps)
        
        ssim_map = numerator / denominator
        ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
        
        return 1.0 - torch.mean(ssim_map)


class PerceptualLoss(nn.Module):
    """
    感知损失 - 基于预训练 VGG-19 提取特征计算 L1 距离
    输入值域: [0, 1] (会自动进行 ImageNet 归一化)
    
    默认提取 relu2_2 和 relu3_3 层特征，权重分别为 1.0 和 1.0
    """
    _vgg_cache = None
    
    def __init__(self, target_layers=None, layer_weights=None):
        super().__init__()
        self.register_buffer('imagenet_mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if target_layers is None:
            target_layers = {'relu2_2': 1.0, 'relu3_3': 1.0}
        self.target_layers = target_layers
        
        self.vgg = self._get_vgg()
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.layer_weights = list(target_layers.values())
        self.layer_names = list(target_layers.keys())
        
        self._register_hooks()
    
    @classmethod
    def _get_vgg(cls):
        if cls._vgg_cache is not None:
            return cls._vgg_cache
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
        except Exception:
            try:
                import torchvision.models as models
                vgg = models.vgg19(pretrained=False).features
                print("  ⚠️ PerceptualLoss: VGG-19 pretrained weights not found, using random init")
            except Exception:
                print("  ⚠️ PerceptualLoss: torchvision not available, returning dummy loss")
                return None
        cls._vgg_cache = vgg
        return vgg
    
    def _register_hooks(self):
        if self.vgg is None:
            return
        
        self._features = {}
        layer_idx_map = {
            'relu1_1': 0, 'relu1_2': 2,
            'relu2_1': 5, 'relu2_2': 7,
            'relu3_1': 10, 'relu3_2': 12, 'relu3_3': 14, 'relu3_4': 16,
            'relu4_1': 19, 'relu4_2': 21, 'relu4_3': 23, 'relu4_4': 25,
            'relu5_1': 28, 'relu5_2': 30, 'relu5_3': 32, 'relu5_4': 34,
        }
        
        def get_hook(name):
            def hook_fn(module, input, output):
                self._features[name] = output
            return hook_fn
        
        for name in self.layer_names:
            if name in layer_idx_map:
                idx = layer_idx_map[name]
                self.vgg[idx].register_forward_hook(get_hook(name))
    
    def forward(self, pred, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        pred = torch.clamp(pred, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)
        
        device = pred.device
        
        if self.imagenet_mean.device != device:
            self.imagenet_mean = self.imagenet_mean.to(device)
            self.imagenet_std = self.imagenet_std.to(device)
        
        if next(self.vgg.parameters()).device != device:
            self.vgg = self.vgg.to(device)
        
        pred_norm = (pred - self.imagenet_mean) / self.imagenet_std
        target_norm = (target - self.imagenet_mean) / self.imagenet_std
        
        pred_norm = pred_norm.float()
        target_norm = target_norm.float()
        
        with torch.autocast(device_type='cuda', enabled=False):
            self.vgg(pred_norm)
            pred_feats = dict(self._features)
            
            self.vgg(target_norm)
            target_feats = dict(self._features)
        
        loss = torch.tensor(0.0, device=pred.device)
        for name, weight in zip(self.layer_names, self.layer_weights):
            if name in pred_feats and name in target_feats:
                feat_loss = F.l1_loss(pred_feats[name], target_feats[name])
                loss = loss + weight * feat_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=pred.device)
        
        return loss


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0, eps=1e-8):
        super().__init__()
        self.max_val = max_val
        self.eps = eps
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps, max=self.max_val)
        target = torch.clamp(target, min=self.eps, max=self.max_val)
        
        mse = F.mse_loss(pred, target)
        mse = torch.clamp(mse, min=self.eps)
        
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        psnr = torch.clamp(psnr, min=0.0, max=100.0)
        return -psnr / 100.0


class FrequencyLoss(nn.Module):
    """
    频域损失 (Frequency Loss)
    计算输出与 GT 的 FFT 幅度差的 log1p L1 距离
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_f = pred.float()
        target_f = target.float()

        with torch.autocast(device_type='cuda', enabled=False):
            pred_f = torch.clamp(pred_f, min=0.0, max=1.0)
            target_f = torch.clamp(target_f, min=0.0, max=1.0)

            pred_fft = torch.fft.rfft2(pred_f, norm='ortho')
            target_fft = torch.fft.rfft2(target_f, norm='ortho')

            pred_amp = torch.abs(pred_fft)
            target_amp = torch.abs(target_fft)
            pred_amp = torch.clamp(pred_amp, min=self.eps, max=1e3)
            target_amp = torch.clamp(target_amp, min=self.eps, max=1e3)

            pred_amp_log = torch.log1p(pred_amp)
            target_amp_log = torch.log1p(target_amp)

            freq_loss = torch.mean(torch.abs(pred_amp_log - target_amp_log))

            if torch.isnan(freq_loss) or torch.isinf(freq_loss):
                freq_loss = torch.tensor(0.0, device=freq_loss.device)

        return freq_loss


class MultiScaleLoss(nn.Module):
    """
    多尺度多阶段损失 - 用于低光图像增强
    
    组合: Charbonnier(主) + SSIM(结构) + Perceptual(感知) + Frequency(频域)
    
    DF-SSM的输出格式：
    outputs_all = [
        [stage1_scale1, stage1_scale2, ...],
        [stage2_scale1, stage2_scale2, ...],
        ...
    ]
    
    推荐权重:
    - Charbonnier: 1.0 (主损失)
    - SSIM: 0.05 (结构相似性)
    - Perceptual: 0.1 (感知质量, VGG特征)
    - Frequency: 0.1 (频域一致性, 帮助全局光照)
    """
    def __init__(self, base_loss=None, stage_weights=None, scale_weights=None,
                 l1_weight=0.0, ssim_weight=0.05, psnr_weight=0.0,
                 frequency_weight=0.1, perceptual_weight=0.0):
        super().__init__()
        self.base_loss = base_loss or CharbonnierLoss()
        self.stage_weights = stage_weights
        self.scale_weights = scale_weights
        self.ssim_loss = SSIMLoss()
        self.frequency_loss = FrequencyLoss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.psnr_weight = psnr_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight
    
    def _match_size(self, x, target_shape):
        if x.shape[2:] == target_shape[2:]:
            return x
        return F.interpolate(x, size=target_shape[2:], mode='bilinear', align_corners=False)
    
    def forward(self, outputs_all, targets):
        num_stages = len(outputs_all)
        num_scales = len(outputs_all[0])
        
        if self.stage_weights is None:
            stage_weights = [1.0] * num_stages
        else:
            stage_weights = self.stage_weights
        
        if self.scale_weights is None:
            scale_weights = [1.0] * num_scales
        else:
            scale_weights = self.scale_weights
        
        if isinstance(targets, torch.Tensor):
            targets_list = []
            for i in range(num_scales):
                if i == 0:
                    targets_list.append(targets)
                else:
                    h, w = targets.shape[2] // (2**i), targets.shape[3] // (2**i)
                    targets_list.append(
                        F.interpolate(targets, size=(h, w), mode='bilinear', align_corners=False)
                    )
        else:
            targets_list = targets
        
        total_loss = 0.0
        loss_dict = {}

        for s_idx, stage_outputs in enumerate(outputs_all):
            stage_loss = 0.0
            for sc_idx, output in enumerate(stage_outputs):
                target = targets_list[sc_idx]
                
                if output.shape != target.shape:
                    output = self._match_size(output, target.shape)
                
                loss = self.base_loss(output, target)

                if sc_idx == 0 and self.ssim_weight > 0:
                    out_clamped = torch.clamp(output, 0.0, 1.0)
                    tgt_clamped = torch.clamp(target, 0.0, 1.0)
                    ssim_l = self.ssim_loss(out_clamped, tgt_clamped)
                    loss = loss + self.ssim_weight * ssim_l

                if sc_idx == 0 and self.frequency_weight > 0:
                    freq_l = self.frequency_loss(output, target)
                    loss = loss + self.frequency_weight * freq_l

                if sc_idx == 0 and self.perceptual_weight > 0 and self.perceptual_loss is not None:
                    out_clamped = torch.clamp(output, 0.0, 1.0)
                    tgt_clamped = torch.clamp(target, 0.0, 1.0)
                    perc_l = self.perceptual_loss(out_clamped, tgt_clamped)
                    loss = loss + self.perceptual_weight * perc_l

                weight = stage_weights[s_idx] * scale_weights[sc_idx]
                weighted_loss = loss * weight
                stage_loss += weighted_loss
                loss_dict[f'stage{s_idx+1}_scale{sc_idx+1}'] = loss.item()

            total_loss += stage_loss
            loss_dict[f'stage{s_idx+1}_total'] = stage_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


def create_loss(loss_type='charbonnier', **kwargs):
    if loss_type == 'charbonnier':
        return CharbonnierLoss(**kwargs)
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'ssim':
        return SSIMLoss(**kwargs)
    elif loss_type == 'perceptual':
        return PerceptualLoss(**kwargs)
    elif loss_type == 'frequency':
        return FrequencyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
