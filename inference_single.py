import argparse
import os
import glob
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models import create_base_net
from utils import load_checkpoint, inference, save_image, bilateral_filter


def find_best_checkpoint(checkpoint_dir):
    """查找验证集PSNR最高的checkpoint"""
    if not os.path.isdir(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        return None
    
    best_psnr = 0
    best_path = None
    
    for cp in checkpoints:
        try:
            cp_name = os.path.basename(cp)
            parts = cp_name.replace('.pth', '').split('_')
            for part in parts:
                if 'psnr' in part.lower():
                    psnr = float(part.split('psnr')[1])
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_path = cp
                    break
        except:
            pass
    
    if best_path is None:
        final_path = os.path.join(checkpoint_dir, 'final_model.pth')
        if os.path.exists(final_path):
            best_path = final_path
        else:
            best_path = checkpoints[-1]
    
    return best_path


def main():
    parser = argparse.ArgumentParser(description='DF-SSM Single Image Inference')
    
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径（默认保存在输入图像同目录）')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoints/final_model.pth', help='模型权重路径')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录，自动查找PSNR最高的权重')
    parser.add_argument('--variant', type=str, default='M-2', choices=['S-1', 'S-2', 'S-3', 'M-1', 'M-2', 'M-3'], help='模型变体')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--tile_size', type=int, default=512, help='分块大小')
    parser.add_argument('--tile_overlap', type=int, default=192, help='重叠区域大小')
    parser.add_argument('--use_bilateral', action='store_true', default=True, help='是否使用双边滤波后处理')
    parser.add_argument('--use_amp', action='store_true', default=True, help='是否使用FP16混合精度推理')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at {args.input}")
        return
    
    if args.checkpoint_dir and os.path.isdir(args.checkpoint_dir):
        best_cp = find_best_checkpoint(args.checkpoint_dir)
        if best_cp:
            args.checkpoint = best_cp
            print(f"Found best checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading DF-SSM-{args.variant} model...")
    model = create_base_net(variant=args.variant, num_supervision_scales=3)
    
    with torch.no_grad():
        model(torch.randn(1, 3, 256, 256))
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    model.eval()
    
    print(f"Processing image: {args.input}")
    img = Image.open(args.input).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0)
    
    print(f"Input size: {img.size}")
    
    output = inference(model, img_tensor, device=device, 
                      tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                      use_amp=args.use_amp)
    
    if args.use_bilateral:
        output = bilateral_filter(output, kernel_size=3, sigma_color=10, sigma_space=8)
    
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_restored.png")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(output, args.output)
    
    print(f"Result saved to: {args.output}")


if __name__ == '__main__':
    main()