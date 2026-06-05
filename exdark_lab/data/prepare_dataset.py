"""
ExDark数据集预处理脚本
=======================
下载并预处理ExDark (Exclusively Dark Image Dataset) 数据集。

功能:
  1. 自动下载ExDark数据集 (或使用本地路径)
  2. 将标注转为YOLO格式
  3. 8:1:1 划分训练/验证/测试集
  4. 生成dataset.yaml

ExDark类别 (12类):
  Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table
"""

import os
import sys
import shutil
import random
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# ExDark 12个类别 (索引0-11)
EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

EXDARK_CLASS_TO_ID = {cls: idx for idx, cls in enumerate(EXDARK_CLASSES)}


def convert_to_yolo(bbox, img_w, img_h):
    """
    将原始标注(左上右下坐标)转为YOLO格式(中心x, 中心y, 宽, 高)
    
    Args:
        bbox: [x1, y1, x2, y2] 左上角右下角坐标
        img_w: 图像宽度
        img_h: 图像高度
    Returns:
        [x_center, y_center, width, height] 归一化后的YOLO格式
    """
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    
    # 裁剪到有效范围
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [x_center, y_center, width, height]


def parse_exdark_annotation(txt_path):
    """
    解析ExDark的原始标注文件
    
    ExDark标注格式: 每行 "x1,y1,x2,y2,class_name"
    
    Args:
        txt_path: 标注文件路径
    Returns:
        annotations: [{'bbox': [x1,y1,x2,y2], 'class_id': int, 'class_name': str}]
    """
    annotations = []
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    x1, y1, x2, y2 = map(float, parts[:4])
                    class_name = parts[4].strip()
                    
                    if class_name in EXDARK_CLASS_TO_ID:
                        annotations.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': EXDARK_CLASS_TO_ID[class_name],
                            'class_name': class_name
                        })
                except (ValueError, IndexError):
                    continue
    
    return annotations


def parse_exdark_annotation_v2(annotation_str):
    """
    解析另一种标注格式: "class_name x1 y1 x2 y2"
    """
    parts = annotation_str.strip().split()
    if len(parts) >= 5:
        class_name = parts[0]
        if class_name in EXDARK_CLASS_TO_ID:
            x1, y1, x2, y2 = map(float, parts[1:5])
            return {
                'bbox': [x1, y1, x2, y2],
                'class_id': EXDARK_CLASS_TO_ID[class_name],
                'class_name': class_name
            }
    return None


def process_exdark_dataset(
    data_root,
    output_root,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    generate_preview=True
):
    """
    处理ExDark数据集的主函数
    
    Args:
        data_root: ExDark原始数据目录
        output_root: 处理后数据集输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        generate_preview: 是否生成预览图像
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # 输出目录结构
    images_dir = {
        'train': output_root / 'images' / 'train',
        'val': output_root / 'images' / 'val',
        'test': output_root / 'images' / 'test',
    }
    labels_dir = {
        'train': output_root / 'labels' / 'train',
        'val': output_root / 'labels' / 'val',
        'test': output_root / 'labels' / 'test',
    }
    
    for d in list(images_dir.values()) + list(labels_dir.values()):
        os.makedirs(d, exist_ok=True)
    
    # --- 收集所有图像文件 ---
    print("[1/5] Collecting ExDark images...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(data_root.rglob(f'*{ext}'))
        all_images.extend(data_root.rglob(f'*{ext.upper()}'))
    
    # 去重并按文件名排序保持一致性
    all_images = sorted(set(all_images))
    
    if len(all_images) == 0:
        print(f"Warning: No images found in {data_root}!")
        print("Creating synthetic test data instead...")
        create_synthetic_test_data(output_root)
        return
    
    print(f"  Found {len(all_images)} images")
    
    # --- 查找对应标注文件 ---
    print("[2/5] Parsing annotations...")
    valid_samples = []
    
    for img_path in tqdm(all_images, desc="Processing"):
        # 尝试多种标注文件路径模式
        possible_anno_paths = [
            img_path.with_suffix('.txt'),
            img_path.parent / (img_path.stem + '.txt'),
            data_root / 'Annotations' / (img_path.stem + '.txt'),
            data_root / 'labels' / (img_path.stem + '.txt'),
            data_root / 'annotation' / (img_path.stem + '.txt'),
        ]
        
        annotations = None
        for anno_path in possible_anno_paths:
            if anno_path.exists():
                annotations = parse_exdark_annotation(anno_path)
                if annotations:
                    break
        
        if annotations and len(annotations) > 0:
            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            valid_samples.append({
                'img_path': str(img_path),
                'img_w': w,
                'img_h': h,
                'annotations': annotations
            })
    
    print(f"  Valid samples: {len(valid_samples)}/{len(all_images)}")
    
    if len(valid_samples) == 0:
        print("No valid samples found. Creating synthetic data...")
        create_synthetic_test_data(output_root)
        return
    
    # --- 划分数据集 ---
    print("[3/5] Splitting dataset (8:1:1)...")
    random.shuffle(valid_samples)
    
    n_total = len(valid_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': valid_samples[:n_train],
        'val': valid_samples[n_train:n_train + n_val],
        'test': valid_samples[n_train + n_val:]
    }
    
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # --- 复制文件并生成YOLO标注 ---
    print("[4/5] Converting to YOLO format...")
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, samples in splits.items():
        out_img_dir = images_dir[split_name]
        out_label_dir = labels_dir[split_name]
        
        for sample in tqdm(samples, desc=f"  {split_name}"):
            # 复制或链接图像
            img_path = sample['img_path']
            img_filename = os.path.basename(img_path)
            dst_img_path = out_img_dir / img_filename
            shutil.copy2(img_path, dst_img_path)
            
            # 生成YOLO格式标注文件
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            dst_label_path = out_label_dir / label_filename
            
            with open(dst_label_path, 'w') as f:
                for ann in sample['annotations']:
                    yolo_bbox = convert_to_yolo(
                        ann['bbox'],
                        sample['img_w'],
                        sample['img_h']
                    )
                    f.write(f"{ann['class_id']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                           f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            stats[split_name] += 1
        
        if generate_preview and split_name == 'test':
            generate_preview_image(dst_img_path, dst_label_path, out_img_dir)
    
    print(f"  Processed: {stats}")
    
    # --- 生成dataset.yaml ---
    print("[5/5] Generating dataset.yaml...")
    generate_dataset_yaml(output_root)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Output: {output_root}")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")


def generate_dataset_yaml(output_root):
    """生成YOLO格式的dataset.yaml配置文件"""
    yaml_content = f"""
# ExDark Dataset Configuration for YOLOv10
# Auto-generated by prepare_dataset.py

path: {output_root}  # dataset root dir
train: images/train  # train images
val: images/val      # val images
test: images/test    # test images

# Classes
nc: 12
names: ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
"""
    
    yaml_path = Path(output_root) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"  Generated: {yaml_path}")


def generate_preview_image(img_path, label_path, output_dir):
    """生成带标注的预览图像"""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, EXDARK_CLASSES[int(cls_id)], (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    preview_path = Path(output_dir) / 'preview.jpg'
    cv2.imwrite(str(preview_path), img)
    print(f"  Preview: {preview_path}")


def create_synthetic_test_data(output_root, n_images=50):
    """
    当真实ExDark数据集不可用时，生成合成测试数据。
    
    Args:
        output_root: 输出目录
        n_images: 生成图像数量
    """
    print("Creating synthetic test images for ExDark...")
    
    for split in ['train', 'val', 'test']:
        img_dir = Path(output_root) / 'images' / split
        label_dir = Path(output_root) / 'labels' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    
    n_per_split = {
        'train': int(n_images * 0.8),
        'val': int(n_images * 0.1),
        'test': int(n_images * 0.1)
    }
    
    for split, count in n_per_split.items():
        img_dir = Path(output_root) / 'images' / split
        label_dir = Path(output_root) / 'labels' / split
        
        for i in tqdm(range(count), desc=f"  Generating {split}"):
            # 创建合成低光图像
            h, w = 640, 640
            img = np.random.randint(10, 50, (h, w, 3), dtype=np.uint8)
            
            # 添加随机物体 (圆形/矩形)
            n_objects = random.randint(1, 4)
            labels = []
            
            for _ in range(n_objects):
                cx = random.randint(50, w-50)
                cy = random.randint(50, h-50)
                obj_w = random.randint(30, 100)
                obj_h = random.randint(30, 100)
                
                # 绘制物体
                color = random.randint(100, 200)
                cv2.rectangle(img, (cx-obj_w//2, cy-obj_h//2),
                            (cx+obj_w//2, cy+obj_h//2), (color, color, color), -1)
                
                class_id = random.randint(0, 11)
                x_c = cx / w
                y_c = cy / h
                bw = obj_w / w
                bh = obj_h / h
                
                labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
            
            # 添加低光效果
            gamma = random.uniform(1.5, 3.0)
            img = ((img / 255.0) ** gamma * 255).astype(np.uint8)
            
            # 添加噪声
            noise = np.random.randn(h, w, 3).astype(np.float32) * 5
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # 保存
            img_path = img_dir / f'synthetic_{i:04d}.jpg'
            label_path = label_dir / f'synthetic_{i:04d}.txt'
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
    
    # 生成dataset.yaml
    generate_dataset_yaml(output_root)
    
    # 生成预览
    test_img_dir = Path(str(output_root)) / 'images' / 'test'
    first_img = next(test_img_dir.glob('*.jpg'), None)
    if first_img:
        first_label = Path(str(first_img).replace('images', 'labels').replace('.jpg', '.txt'))
        if first_label.exists():
            generate_preview_image(first_img, first_label, test_img_dir)
    
    print(f"  Generated {n_images} synthetic images")
    print(f"  Output: {output_root}")


def download_exdark(url=None):
    """
    下载ExDark数据集的提示
    
    ExDark数据集需要从原作者处获取:
    https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
    """
    print("=" * 60)
    print("ExDark Dataset Download Guide")
    print("=" * 60)
    print()
    print("ExDark Dataset is available at:")
    print("  https://github.com/cs-chan/Exclusively-Dark-Image-Dataset")
    print()
    print("Option 1: Download from Kaggle")
    print("  https://www.kaggle.com/datasets/duttakolkata/exdark-dataset")
    print()
    print("Option 2: Download from original source")
    print("  git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.git")
    print()
    print("After downloading, extract to ./data/ExDark/")
    print("Then run: python data/prepare_dataset.py --data_root ./data/ExDark")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ExDark Dataset Preparation')
    parser.add_argument('--data_root', type=str, default=None,
                       help='ExDark原始数据路径 (留空则生成合成数据)')
    parser.add_argument('--output_root', type=str, default='./data/processed',
                       help='处理后数据输出路径')
    parser.add_argument('--synthetic', action='store_true',
                       help='强制生成合成数据')
    parser.add_argument('--n_synthetic', type=int, default=100,
                       help='合成数据数量')
    parser.add_argument('--download_guide', action='store_true',
                       help='显示下载指引')
    
    args = parser.parse_args()
    
    if args.download_guide:
        download_exdark()
    elif args.synthetic or args.data_root is None:
        create_synthetic_test_data(args.output_root, n_images=args.n_synthetic)
    else:
        process_exdark_dataset(args.data_root, args.output_root)
