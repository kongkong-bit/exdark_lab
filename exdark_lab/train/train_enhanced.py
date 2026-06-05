"""
实验2: 传统增强预处理 + YOLOv10n 训练
======================================
使用CLAHE、直方图均衡化等传统图像增强方法预处理后训练。
验证传统方法的局限性。

支持的增强方法:
  1. CLAHE (对比度受限自适应直方图均衡化)
  2. 直方图均衡化 (HE)
  3. 伽马校正
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from utils.helpers import ExperimentTracker, setup_seed, get_device


def apply_clahe(img):
    """
    CLAHE增强
    
    Args:
        img: [H, W, 3] BGR图像 uint8
    Returns:
        enhanced: 增强后的BGR图像
    """
    # 转换到LAB空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # 合并
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_histogram_equalization(img):
    """
    直方图均衡化增强
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(img, gamma=0.5):
    """
    伽马校正增强
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_msrcr(img, scales=[15, 80, 250]):
    """
    多尺度Retinex (MSRCR) 增强 - 使用OpenCV实现
    """
    img_float = img.astype(np.float32) + 1.0
    h, w = img.shape[:2]
    
    retinex = np.zeros_like(img_float)
    for sigma in scales:
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        retinex += np.log(img_float) - np.log(blur + 1.0)
    
    retinex = retinex / len(scales)
    
    # 颜色恢复
    alpha = 128
    img_sum = np.sum(img_float, axis=2, keepdims=True) + 1
    cr = np.log(alpha * img_float / img_sum)
    cr = np.clip(cr, 0, 1)
    
    result = cr * retinex
    
    # 归一化到[0,255]
    for c in range(3):
        channel = result[:, :, c]
        min_v, max_v = channel.min(), channel.max()
        if max_v > min_v:
            result[:, :, c] = (channel - min_v) / (max_v - min_v) * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_dataset(data_root, method='clahe'):
    """
    对整个数据集的图像进行增强预处理
    
    Args:
        data_root: 数据集根目录
        method: 增强方法 ('clahe', 'histeq', 'gamma', 'msrcr')
    """
    data_root = Path(data_root)
    
    # 创建增强后数据集目录
    enhanced_root = data_root.parent / f'{data_root.name}_enhanced_{method}'
    
    for split in ['train', 'val', 'test']:
        src_img_dir = data_root / 'images' / split
        dst_img_dir = enhanced_root / 'images' / split
        dst_label_dir = enhanced_root / 'labels' / split
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制标注文件 (保持不变)
        src_label_dir = data_root / 'labels' / split
        if src_label_dir.exists():
            for label_file in src_label_dir.glob('*.txt'):
                import shutil
                shutil.copy2(label_file, dst_label_dir / label_file.name)
        
        if not src_img_dir.exists():
            continue
        
        # 增强所有图像
        img_paths = list(src_img_dir.glob('*.*'))
        
        def process_image(img_path):
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            if method == 'clahe':
                enhanced = apply_clahe(img)
            elif method == 'histeq':
                enhanced = apply_histogram_equalization(img)
            elif method == 'gamma':
                enhanced = apply_gamma_correction(img, gamma=0.5)
            elif method == 'msrcr':
                enhanced = apply_msrcr(img)
            else:
                enhanced = img
            
            dst_path = dst_img_dir / img_path.name
            cv2.imwrite(str(dst_path), enhanced)
            return str(dst_path)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(
                executor.map(process_image, img_paths),
                total=len(img_paths),
                desc=f"  Enhancing {split}"
            ))
    
    print(f"Enhanced dataset saved: {enhanced_root}")
    return enhanced_root


def train_with_enhancement(config_path='config.yaml', enhance_method='clahe'):
    """
    使用增强预处理的数据集训练YOLOv10n
    
    Args:
        config_path: 配置文件路径
        enhance_method: 增强方法
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    data_root = Path(config['data'].get('output', './data/processed'))
    
    # 增强数据集
    print(f"\nApplying {enhance_method} enhancement...")
    enhanced_root = enhance_dataset(data_root, method=enhance_method)
    
    # 创建增强后的dataset.yaml
    from data.prepare_dataset import generate_dataset_yaml
    generate_dataset_yaml(enhanced_root)
    enhanced_data_yaml = enhanced_root / 'dataset.yaml'
    
    print("=" * 60)
    print(f"Experiment 2: Traditional Enhancement ({enhance_method}) + YOLOv10n")
    print("=" * 60)
    print(f"Enhanced dataset: {enhanced_root}")
    
    # 训练
    try:
        model = YOLO('yolov10n.pt')
        
        results = model.train(
            data=str(enhanced_data_yaml),
            epochs=config['train'].get('epochs', 100),
            batch=config['train'].get('batch_size', 16),
            imgsz=config['model'].get('img_size', 640),
            lr0=config['train'].get('lr', 0.001),
            device=device,
            workers=config['train'].get('workers', 4),
            project=str(Path(config['logging']['save_dir']) / 'models'),
            name=f'enhanced_{enhance_method}',
            exist_ok=True,
            pretrained=True,
            optimizer=config['train'].get('optimizer', 'AdamW'),
            warmup_epochs=config['train'].get('warmup_epochs', 3),
            amp=config['train'].get('amp', True),
        )
        
        # 记录实验结果
        tracker = ExperimentTracker(Path(config['logging']['save_dir']) / 'logs')
        tracker.log_experiment(
            name=f'enhanced_{enhance_method}',
            config=config,
            metrics={'mAP50': results.results_dict.get('metrics/mAP50(B)', 0)},
            model_path=str(enhanced_root)
        )
        
        return results
    
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Traditional Enhancement + YOLOv10n')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='clahe',
                       choices=['clahe', 'histeq', 'gamma', 'msrcr'])
    parser.add_argument('--simulate', action='store_true', help='Run simulation mode')
    
    args = parser.parse_args()
    
    if args.simulate:
        print("Simulation mode: Testing enhancement on synthetic data...")
        from data.prepare_dataset import create_synthetic_test_data
        create_synthetic_test_data('./data/processed', n_images=20)
        
        test_img = cv2.imread('./data/processed/images/train/synthetic_0000.jpg')
        if test_img is not None:
            enhanced = apply_clahe(test_img)
            print(f"CLAHE test: Input {test_img.shape} -> Output {enhanced.shape}")
            
            enhanced_hist = apply_histogram_equalization(test_img)
            print(f"HistEq test: Input {test_img.shape} -> Output {enhanced_hist.shape}")
            
            enhanced_msrcr = apply_msrcr(test_img)
            print(f"MSRCR test: Input {test_img.shape} -> Output {enhanced_msrcr.shape}")
        
        print("\n✓ Enhancement methods verified!")
        return
    
    results = train_with_enhancement(args.config, args.method)
    if results is None:
        print("\nTraining failed. Use --simulate for test mode.")


if __name__ == '__main__':
    main()
