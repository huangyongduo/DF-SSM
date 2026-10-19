"""
数据集定义 - DF-SSM PyTorch实现
包含：通用图像恢复数据集、数据增强等
"""

import os
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ★ 支持的图像格式 (VE-LOL-L-Cap 数据集通常是 jpg 格式)
SUPPORTED_IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _scan_images(directory, extensions=SUPPORTED_IMG_EXTENSIONS):
    """
    ★ 扫描目录下所有支持格式的图像文件 (不限于单一格式)
    解决 VE-LOL-L-Cap 数据集为 jpg 格式但代码只搜索 png 的问题
    """
    if not os.path.isdir(directory):
        return []
    files = []
    for ext in extensions:
        # 大小写不敏感搜索 (Windows 不敏感, Linux 敏感)
        files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    # 去重并排序
    files = sorted(list(set(files)))
    return files


def _get_filename_stem(path):
    """获取文件名(不含扩展名),用于 input/target 配对"""
    return os.path.splitext(os.path.basename(path))[0]


class ImageRestorationDataset(Dataset):
    """
    通用图像恢复数据集
    支持去噪、去模糊、去雨、去雾、增强等任务

    ★ 渐进式训练支持 (Progressive Training):
    - 通过 set_epoch() 方法动态调整 patch_size
    - 训练初期用小 patch (128) 加速收敛, 后期用大 patch (256/512) 提升细节

    ★ 多格式支持:
    - 自动扫描 png/jpg/jpeg/bmp/tif/tiff/webp 等多种格式
    - 解决 VE-LOL-L-Cap 数据集为 jpg 但代码只搜 png 的问题

    支持的目录结构:
        1. VE-LOL-L-Cap 格式:
            root/
                VE-LOL-L-Cap-Low_train/    (低光照训练输入)
                VE-LOL-L-Cap-Normal_train/ (正常光照训练目标)
                VE-LOL-L-Cap-Low_test/     (低光照测试输入)
                VE-LOL-L-Cap-Normal_test/  (正常光照测试目标)
        2. 标准格式:
            root/
                input/    # 输入图像
                target/   # 目标图像
        3. 自定义子文件夹 (通过 input_folder/target_folder 指定)
    """
    def __init__(
        self,
        root_dir,
        patch_size=256,
        is_train=True,
        augment=True,
        input_folder='input',
        target_folder='target',
        img_format=None,  # ★ None=自动扫描所有支持格式; 指定则只搜该格式
        enable_low_light_aug=True,
        progressive_schedule=None,  # ★ 渐进式训练: [(epoch_threshold, patch_size), ...]
        no_crop=False,  # ★ True=返回整图, 不做任何裁剪 (用于 full-image 评估)
    ):
        super().__init__()
        self.root_dir = root_dir
        self.base_patch_size = patch_size
        self.patch_size = patch_size  # 当前 patch_size (会随 epoch 变化)
        self.is_train = is_train
        self.augment = augment and is_train
        self.enable_low_light_aug = enable_low_light_aug
        self.progressive_schedule = progressive_schedule
        self.current_epoch = 0
        self.no_crop = no_crop  # ★ full-image 模式

        # ★ 如果指定了单一格式, 只搜该格式; 否则搜所有支持格式
        if img_format is not None:
            # 兼容用户传入 'png' 或 '.png'
            ext = img_format if img_format.startswith('.') else f'.{img_format}'
            extensions = (ext.lower(), ext.upper())
        else:
            extensions = SUPPORTED_IMG_EXTENSIONS

        # ========== 自动探测目录结构 ==========
        checked_paths = []  # 记录检查过的路径, 用于错误提示
        input_dir = None
        target_dir = None

        # 1. 尝试 VE-LOL-L-Cap 格式
        if is_train:
            ve_lol_input = os.path.join(root_dir, 'VE-LOL-L-Cap-Low_train')
            ve_lol_target = os.path.join(root_dir, 'VE-LOL-L-Cap-Normal_train')
        else:
            ve_lol_input = os.path.join(root_dir, 'VE-LOL-L-Cap-Low_test')
            ve_lol_target = os.path.join(root_dir, 'VE-LOL-L-Cap-Normal_test')
        checked_paths.append(('VE-LOL-L-Cap', ve_lol_input, ve_lol_target))

        if os.path.isdir(ve_lol_input) and os.path.isdir(ve_lol_target):
            input_dir = ve_lol_input
            target_dir = ve_lol_target

        # 1b. ★ 尝试 LOL-v2 格式: train/low + train/high, test/low + test/high
        if input_dir is None:
            split_name = 'train' if is_train else 'test'
            lol_v2_input = os.path.join(root_dir, split_name, 'low')
            lol_v2_target = os.path.join(root_dir, split_name, 'high')
            checked_paths.append(('LOL-v2', lol_v2_input, lol_v2_target))
            if os.path.isdir(lol_v2_input) and os.path.isdir(lol_v2_target):
                input_dir = lol_v2_input
                target_dir = lol_v2_target

        # 1d. ★ LOL-v2-Real 官方结构: Train/Low + Train/Normal, Test/Low + Test/Normal
        #   (GT 目录名是 Normal 而非 high, 输入目录名是 Low 而非 low, split 名首字母大写)
        if input_dir is None:
            split_real = 'Train' if is_train else 'Test'
            real_input = os.path.join(root_dir, split_real, 'Low')
            real_target = os.path.join(root_dir, split_real, 'Normal')
            checked_paths.append(('LOL-v2-Real', real_input, real_target))
            if os.path.isdir(real_input) and os.path.isdir(real_target):
                input_dir = real_input
                target_dir = real_target

        # 1c. ★ 尝试经典 LOL-v1 格式: {root}/low + {root}/high
        #   (如 our485/low + our485/high, eval15/low + eval15/high)
        if input_dir is None:
            lol_v1_input = os.path.join(root_dir, 'low')
            lol_v1_target = os.path.join(root_dir, 'high')
            checked_paths.append(('LOL-v1 low/high', lol_v1_input, lol_v1_target))
            if os.path.isdir(lol_v1_input) and os.path.isdir(lol_v1_target):
                input_dir = lol_v1_input
                target_dir = lol_v1_target

        # 1e. ★ LOL-v2 的首字母大写 split 目录: {root}/Low + {root}/Normal
        #   用于 root_dir 直接指向 split 目录的场景, 如:
        #     --val_dir datasets/LOLv2/Synthetic/Test  (内含 Test/Low + Test/Normal)
        #   若缺少此规则, 会掉到"单子文件夹兜底", 导致 input_dir == target_dir
        #   (输入被当成 GT, 验证 PSNR 会异常低至 ~10dB)
        if input_dir is None:
            cap_input = os.path.join(root_dir, 'Low')
            cap_target = os.path.join(root_dir, 'Normal')
            checked_paths.append(('LOL-v2 Low/Normal (split dir)', cap_input, cap_target))
            if os.path.isdir(cap_input) and os.path.isdir(cap_target):
                input_dir = cap_input
                target_dir = cap_target

        # 2. 尝试标准 input/target 格式
        if input_dir is None:
            standard_input = os.path.join(root_dir, input_folder)
            standard_target = os.path.join(root_dir, target_folder)
            checked_paths.append(('standard input/target', standard_input, standard_target))
            if os.path.isdir(standard_input) and os.path.isdir(standard_target):
                input_dir = standard_input
                target_dir = standard_target

        # 4. 尝试直接在 root_dir 下找图像 (input=target=root_dir)
        if input_dir is None:
            root_imgs = _scan_images(root_dir, extensions)
            checked_paths.append(('root_dir direct', root_dir, root_dir))
            if len(root_imgs) > 0:
                input_dir = root_dir
                target_dir = root_dir  # ★ 单文件夹模式: input == target

        # 5. ★ 单文件夹模式: 如果只有一个子文件夹, 且里面有图像
        # 这处理 VE-LOL-L-Cap-ME 等单文件夹结构, 用于自监督学习 (input=target)
        if input_dir is None:
            subdirs = []
            for item in sorted(os.listdir(root_dir)):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    subdirs.append(item_path)
            
            for subdir in subdirs:
                subdir_imgs = _scan_images(subdir, extensions)
                if len(subdir_imgs) > 0:
                    checked_paths.append(('single subdir', subdir, subdir))
                    if input_dir is None:
                        input_dir = subdir
                        target_dir = subdir  # ★ 单文件夹模式: input == target
                    else:
                        break  # 如果有多个子文件夹, 用第二个作为 target (但这可能不是预期行为)

        # 6. 全部失败, 使用默认路径 (后续会触发 assert 并显示详细错误)
        if input_dir is None:
            input_dir = os.path.join(root_dir, input_folder)
            target_dir = os.path.join(root_dir, target_folder)

        self.input_dir = input_dir
        self.target_dir = target_dir

        # ========== 扫描图像文件 (支持多格式) ==========
        self.input_files = _scan_images(self.input_dir, extensions)
        self.target_files = _scan_images(self.target_dir, extensions)

        # ========== 文件配对 (基于文件名 stem, 不依赖扩展名) ==========
        # 解决 input 为 jpg 但 target 为 png 等格式不一致问题
        if len(self.input_files) > 0 and len(self.target_files) > 0:
            self.input_files, self.target_files = self._pair_files_by_stem(
                self.input_files, self.target_files
            )

        # ========== 详细的错误提示 ==========
        if len(self.input_files) == 0 or len(self.target_files) == 0:
            # 构建详细的错误信息
            err_msg = [
                f"\n{'='*70}",
                f"❌ 数据集加载失败: 未找到图像文件",
                f"{'='*70}",
                f"root_dir: {root_dir}",
                f"is_train: {is_train}",
                f"扫描的图像格式: {extensions}",
                f"当前使用的路径:",
                f"  input_dir:  {self.input_dir} ({len(self.input_files)} 张图像)",
                f"  target_dir: {self.target_dir} ({len(self.target_files)} 张图像)",
                f"",
                f"已尝试的目录结构:",
            ]
            for name, p_in, p_tgt in checked_paths:
                in_exists = '✅' if os.path.isdir(p_in) else '❌'
                tgt_exists = '✅' if os.path.isdir(p_tgt) else '❌'
                err_msg.append(f"  [{name}] input:{in_exists} {p_in}")
                err_msg.append(f"             target:{tgt_exists} {p_tgt}")

            err_msg.append(f"")
            err_msg.append(f"目录实际内容 (root_dir 下):")
            if os.path.isdir(root_dir):
                for item in sorted(os.listdir(root_dir)):
                    item_path = os.path.join(root_dir, item)
                    if os.path.isdir(item_path):
                        n_files = len(_scan_images(item_path, extensions))
                        err_msg.append(f"  📁 {item}/ ({n_files} 张图像)")
                    else:
                        err_msg.append(f"  📄 {item}")
            else:
                err_msg.append(f"  ❌ root_dir 不存在!")

            err_msg.append(f"")
            err_msg.append(f"建议:")
            err_msg.append(f"  1. 检查 --train_dir / --val_dir 路径是否正确")
            err_msg.append(f"  2. 确认数据集目录结构符合上述任一格式")
            err_msg.append(f"  3. VE-LOL-L-Cap 数据集应包含:")
            err_msg.append(f"     VE-LOL-L-Cap-Low_train/ 和 VE-LOL-L-Cap-Normal_train/")
            err_msg.append(f"  4. LOL-v2 数据集应包含:")
            err_msg.append(f"     train/low/ + train/high/  和  test/low/ + test/high/")
            err_msg.append(f"  5. 如需指定图像格式, 可传 img_format='jpg'")
            err_msg.append(f"{'='*70}")

            raise FileNotFoundError('\n'.join(err_msg))

        assert len(self.input_files) == len(self.target_files), \
            f"Input ({len(self.input_files)}) and target ({len(self.target_files)}) count mismatch " \
            f"after pairing by filename stem"

    def set_augment(self, enabled):
        """动态开关数据增强（用于验证集禁用增强）"""
        self.augment = enabled and self.is_train

    @staticmethod
    def _pair_files_by_stem(input_files, target_files):
        """
        ★ 基于文件名进行 input/target 配对
        支持多种命名风格:
        1. 完全相同 stem: img_001.jpg ↔ img_001.png
        2. 相同数字索引: low00101.png ↔ normal00101.png (前缀不同但数字部分相同)
        """
        import re
        
        # 构建 target 映射
        # 先尝试基于完整 stem
        target_full_map = {}
        # 再尝试基于数字索引 (提取文件名中的数字部分)
        target_num_map = {}
        
        for tf in target_files:
            stem = _get_filename_stem(tf)
            target_full_map[stem] = tf
            
            # 提取数字部分 (如 low00101 -> 00101, img_001 -> 001)
            nums = re.findall(r'\d+', stem)
            if nums:
                # 用所有数字拼接作为索引 (处理如 'low00101' 这种连续数字)
                num_idx = ''.join(nums)
                target_num_map[num_idx] = tf
        
        paired_input = []
        paired_target = []
        missing_targets = []
        
        for inp in input_files:
            stem = _get_filename_stem(inp)
            
            # 策略 1: 基于完整 stem 配对
            if stem in target_full_map:
                paired_input.append(inp)
                paired_target.append(target_full_map[stem])
                continue
            
            # 策略 2: 基于数字索引配对 (处理 low00101 vs normal00101)
            nums = re.findall(r'\d+', stem)
            if nums:
                num_idx = ''.join(nums)
                if num_idx in target_num_map:
                    paired_input.append(inp)
                    paired_target.append(target_num_map[num_idx])
                    continue
            
            # 策略 3: 都不匹配
            missing_targets.append(stem)
        
        if len(missing_targets) > 0 and len(missing_targets) < 10:
            print(f"  ⚠️ {len(missing_targets)} 个 input 找不到对应 target: {missing_targets[:5]}")
        elif len(missing_targets) >= 10:
            print(f"  ⚠️ {len(missing_targets)} 个 input 找不到对应 target (前5个): {missing_targets[:5]}")
        
        return paired_input, paired_target

    def set_epoch(self, epoch):
        """
        ★ 渐进式训练: 根据 epoch 动态调整 patch_size
        progressive_schedule 格式: [(epoch_threshold, patch_size), ...]
        例如: [(0, 128), (100, 256), (300, 512)] 表示
              epoch 0-99 用 128, 100-299 用 256, 300+ 用 512
        """
        self.current_epoch = epoch
        if self.progressive_schedule is not None and self.is_train:
            # 找到当前 epoch 对应的 patch_size
            new_ps = self.base_patch_size
            for threshold, ps in self.progressive_schedule:
                if epoch >= threshold:
                    new_ps = ps
            if new_ps != self.patch_size:
                self.patch_size = new_ps

    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx]).convert('RGB')
        target_img = Image.open(self.target_files[idx]).convert('RGB')

        # ★ full-image 模式: 不做任何裁剪, 直接返回整图
        #   (训练时 no_crop=True 且 augment=True 仍需施加增强)
        if self.no_crop:
            if self.augment:
                input_img, target_img = self._augment(input_img, target_img)
            input_tensor = TF.to_tensor(input_img)
            target_tensor = TF.to_tensor(target_img)
            return input_tensor, target_tensor

        if self.is_train:
            input_img, target_img = self._random_crop(input_img, target_img)
        else:
            input_img, target_img = self._center_crop(input_img, target_img)

        if self.augment:
            input_img, target_img = self._augment(input_img, target_img)

        # 转为Tensor
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        return input_tensor, target_tensor
    
    def _center_crop(self, input_img, target_img):
        """中心裁剪patch"""
        w, h = input_img.size
        ps = self.patch_size
        
        if w < ps or h < ps:
            input_img = input_img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            target_img = target_img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = input_img.size
        
        x = (w - ps) // 2
        y = (h - ps) // 2
        
        input_patch = input_img.crop((x, y, x + ps, y + ps))
        target_patch = target_img.crop((x, y, x + ps, y + ps))
        
        return input_patch, target_patch
    
    def _random_crop(self, input_img, target_img):
        """随机裁剪patch"""
        w, h = input_img.size
        ps = self.patch_size
        
        if w < ps or h < ps:
            # 如果图像太小，调整大小
            input_img = input_img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            target_img = target_img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = input_img.size
        
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        
        input_patch = input_img.crop((x, y, x + ps, y + ps))
        target_patch = target_img.crop((x, y, x + ps, y + ps))
        
        return input_patch, target_patch
    
    def _augment(self, input_img, target_img):
        """
        ★ 配对数据增强 - 所有空间变换对 input 和 GT 完全一致 (像素对齐)

        安全增强 (p=0.5):
          - 随机水平翻转
          - 随机垂直翻转
          - 随机 90/180/270 度旋转

        可选增强 (p=0.3):
          - 轻度颜色抖动 (仅施加于 input, 不施加于 GT)
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
        """
        # ========== 安全空间变换 (input 和 GT 完全同步) ==========
        # 随机水平翻转 (p=0.5)
        if random.random() < 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        # 随机垂直翻转 (p=0.5)
        if random.random() < 0.5:
            input_img = TF.vflip(input_img)
            target_img = TF.vflip(target_img)

        # 随机旋转 (p=0.5)
        # ★ 全图模式下只能用 180°(保形): 90/270 会把 600x400 转成 400x600,
        #   同一 batch 内形状不一致 → DataLoader collate 失败
        if random.random() < 0.5:
            angle = 180 if self.no_crop else random.choice([90, 180, 270])
            input_img = TF.rotate(input_img, angle)
            target_img = TF.rotate(target_img, angle)

        # ========== 配对曝光增强 (input 和 GT 同步) ==========
        # 曝光增益 + gamma 会改变像素的"绝对值",
        # ★ 必须对 input 和 GT 施加完全相同的系数/曲线, 否则监督信号被破坏
        #   (低光 input 被强制提亮到接近 GT, 同时 GT 不变 → SSIM/PSNR 直接崩)
        if self.enable_low_light_aug and random.random() < 0.5:
            exposure = random.uniform(0.7, 1.3)
            gamma_factor = random.uniform(0.7, 1.3)
            input_img  = ImageEnhance.Brightness(input_img).enhance(exposure)
            target_img = ImageEnhance.Brightness(target_img).enhance(exposure)
            input_img  = TF.adjust_gamma(input_img,  gamma_factor, gain=1.0)
            target_img = TF.adjust_gamma(target_img, gamma_factor, gain=1.0)

        # ========== 仅 input 退化增强 (向"更差"方向扰动 input, GT 保持干净) ==========
        # 这些项模拟的是"真实拍摄条件下低光端额外的缺陷",
        # 它们只让 input 更差, 不改相对亮度/对比度的量级, 因此不会误伤监督.
        if self.enable_low_light_aug and random.random() < 0.5:
            # 1. 轻微对比度 + 饱和度抖动 (范围较小, 不改变整体亮度)
            contrast_factor = random.uniform(0.9, 1.1)
            saturation_factor = random.uniform(0.9, 1.1)
            hue_shift = random.uniform(-0.01, 0.01)
            input_img = TF.adjust_contrast(input_img, contrast_factor)
            input_img = TF.adjust_saturation(input_img, saturation_factor)
            input_img = TF.adjust_hue(input_img, hue_shift)
            # 2. 色温/白平衡扰动 (R/B 通道独立缩放, 模拟光源色偏)
            r_gain = random.uniform(0.92, 1.08)
            b_gain = random.uniform(0.92, 1.08)
            wb_arr = np.asarray(input_img).astype(np.float32)
            wb_arr[..., 0] *= r_gain
            wb_arr[..., 2] *= b_gain
            wb_arr = np.clip(wb_arr, 0, 255)
            input_img = Image.fromarray(wb_arr.astype(np.uint8))
            # 3. 轻高斯噪声 (模拟低光传感器读出噪声)
            noise_std = random.uniform(0.0, 0.01)
            if noise_std > 0.001:
                arr = np.asarray(input_img).astype(np.float32)
                arr = np.clip(arr + np.random.normal(0.0, noise_std * 255.0, arr.shape), 0, 255)
                input_img = Image.fromarray(arr.astype(np.uint8))

        return input_img, target_img


class PairedImageFolder(Dataset):
    """
    配对图像文件夹数据集
    适用于GoPro、Rain13k等标准数据集格式

    ★ 支持多格式图像扫描 (png/jpg/jpeg/bmp/tif/tiff/webp)
    """
    def __init__(
        self,
        input_dir,
        target_dir,
        patch_size=256,
        is_train=True,
        augment=True,
        img_format=None  # ★ None=自动扫描所有支持格式
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment and is_train

        # ★ 多格式扫描
        if img_format is not None:
            ext = img_format if img_format.startswith('.') else f'.{img_format}'
            extensions = (ext.lower(), ext.upper())
        else:
            extensions = SUPPORTED_IMG_EXTENSIONS

        self.input_files = _scan_images(input_dir, extensions)
        self.target_files = _scan_images(target_dir, extensions)

        # 基于文件名 stem 配对
        if len(self.input_files) > 0 and len(self.target_files) > 0:
            self.input_files, self.target_files = ImageRestorationDataset._pair_files_by_stem(
                self.input_files, self.target_files
            )

        assert len(self.input_files) > 0, \
            f"No images found in input_dir: {input_dir} (scanned formats: {extensions})"
        assert len(self.input_files) == len(self.target_files), \
            f"Input ({len(self.input_files)}) and target ({len(self.target_files)}) count mismatch"
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx]).convert('RGB')
        target_img = Image.open(self.target_files[idx]).convert('RGB')
        
        if self.is_train:
            # 随机裁剪
            i, j, h, w = self._get_random_crop_params(input_img)
            input_img = TF.crop(input_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            
            if self.augment:
                input_img, target_img = self._augment(input_img, target_img)
        
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)
        
        return input_tensor, target_tensor
    
    def _get_random_crop_params(self, img):
        w, h = img.size
        th, tw = self.patch_size, self.patch_size
        
        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Image size ({h}, {w}) smaller than patch size ({th}, {tw})")
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return i, j, th, tw
    
    def _augment(self, img1, img2):
        # 随机水平翻转 (p=0.5)
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # 随机垂直翻转 (p=0.5)
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # 随机 90/180/270 度旋转 (p=0.5)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)

        return img1, img2


def create_dataloader(
    dataset,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4
):
    """
    创建DataLoader的工具函数

    ★ 性能优化:
    - persistent_workers: 避免每 epoch 重新 fork worker (省启动开销)
    - prefetch_factor: 每个 worker 预取 4 batch, 保证 GPU 不饿死
    """
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    # prefetch_factor / persistent_workers 仅在 num_workers>0 时有效
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = prefetch_factor
    loader = torch.utils.data.DataLoader(**kwargs)
    return loader


class MultiDatasetChain(Dataset):
    """
    ★ P7: 多数据集联合训练
    将多个数据集串联, 训练时随机从各数据集中采样, 保持 batch 内混合

    用法:
        dataset = MultiDatasetChain([
            ImageRestorationDataset(root_dir='path/to/lol_v2', ...),
            ImageRestorationDataset(root_dir='path/to/sid_sony', ...),
        ], weights=[0.5, 0.5])
    """
    def __init__(self, datasets, weights=None, length_multiplier=1.0):
        """
        Args:
            datasets: 数据集列表
            weights: 各数据集的采样权重 (None=按数据量比例)
            length_multiplier: 总长度的倍率 (1.0=取最大数据集长度,
                               2.0=两倍, 实现更多 epoch 内采样)
        """
        super().__init__()
        self.datasets = datasets
        self.num_datasets = len(datasets)

        # 各数据集长度
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.max_length = max(self.dataset_lengths)
        self.total_length = int(self.max_length * length_multiplier)

        # 采样权重
        if weights is None:
            # 按数据量比例分配权重
            total = sum(self.dataset_lengths)
            self.weights = [l / total for l in self.dataset_lengths]
        else:
            assert len(weights) == self.num_datasets
            self.weights = weights

        # 预计算累积权重 (用于加权随机采样)
        self.cumulative_weights = []
        cum = 0.0
        for w in self.weights:
            cum += w
            self.cumulative_weights.append(cum)

        print(f"  ★ MultiDatasetChain: {self.num_datasets} datasets, "
              f"lengths={self.dataset_lengths}, weights={self.weights}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # ★ 加权随机选择数据集
        r = random.random()
        ds_idx = 0
        for i, cw in enumerate(self.cumulative_weights):
            if r <= cw:
                ds_idx = i
                break

        # 从选中的数据集中随机采样一个样本
        local_idx = random.randint(0, self.dataset_lengths[ds_idx] - 1)
        return self.datasets[ds_idx][local_idx]

    def set_epoch(self, epoch):
        """渐进式训练: 传递 epoch 到所有子数据集"""
        for ds in self.datasets:
            _ds = ds.dataset if hasattr(ds, 'dataset') else ds
            if hasattr(_ds, 'set_epoch'):
                _ds.set_epoch(epoch)


def split_train_val_dataset(
    root_dir,
    val_ratio=0.1,
    seed=42,
    patch_size=256,
    **kwargs
):
    """
    自动将一个数据集划分为训练集和验证集。
    当 val_dir 未提供或与 train_dir 相同时，使用此函数自动划分。

    ★ 修复: 训练和验证集都使用同一个底层数据集(is_train=True),
      仅通过 set_augment() 控制是否增强, 避免索引越界问题。

    Args:
        root_dir: 数据集根目录
        val_ratio: 验证集比例 (0.05 ~ 0.3 推荐)
        seed: 随机种子（保证可复现）
        patch_size: 裁剪patch大小
        **kwargs: 其余传递给 ImageRestorationDataset 的参数

    Returns:
        (train_dataset, val_dataset)
    """
    import random
    from torch.utils.data import Subset

    full_dataset = ImageRestorationDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        is_train=True,
        augment=True,
        **kwargs
    )

    total_size = len(full_dataset)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size

    if val_size == 0:
        print(f"  ⚠️ 数据集太小 ({total_size} 张)，跳过验证集划分")
        return full_dataset, None

    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)

    val_base_dataset = ImageRestorationDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        is_train=True,
        augment=True,
        **kwargs
    )
    val_base_dataset.set_augment(False)
    val_dataset = Subset(val_base_dataset, val_indices)

    print(f"  ★ 自动划分: 训练 {train_size} 张, 验证 {val_size} 张 "
          f"(val_ratio={val_ratio})")

    return train_dataset, val_dataset
