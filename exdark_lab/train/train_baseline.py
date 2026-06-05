"""
实验1: Baseline YOLOv10n 训练
==============================
直接在原始低光图像上训练YOLOv10n检测器。
作为性能基准。

使用方式:
  python train/train_baseline.py --data ./data/processed/dataset.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ultralytics import YOLO
from utils.helpers import ExperimentTracker, setup_seed, get_device


def train_baseline(config_path='config.yaml'):
    """
    Baseline训练: 直接使用YOLOv10n在原始低光图像上训练
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子和设备
    setup_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # 数据集配置
    data_yaml = config['data'].get('dataset_yaml', './data/processed/dataset.yaml')
    if not os.path.exists(data_yaml):
        print(f"Dataset config not found: {data_yaml}")
        print("Run 'python data/prepare_dataset.py --synthetic' to generate test data first.")
        return None
    
    # 训练参数
    model_name = config['model'].get('name', 'yolov10n')
    epochs = config['train'].get('epochs', 100)
    batch_size = config['train'].get('batch_size', 16)
    img_size = config['model'].get('img_size', 640)
    lr = config['train'].get('lr', 0.001)
    workers = config['train'].get('workers', 4)
    
    # 保存路径
    save_dir = Path(config['logging'].get('save_dir', './results'))
    exp_name = config['logging'].get('experiment_name', 'baseline_yolov10n')
    
    print("=" * 60)
    print("Experiment 1: Baseline YOLOv10n Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Image size: {img_size}")
    print("=" * 60)
    
    try:
        # 尝试加载YOLOv10模型
        # YOLOv10使用与YOLOv8相同的ultralytics API
        model = YOLO(f'{model_name}.pt')  # 自动下载预训练权重
        
        # 开始训练
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr,
            device=device,
            workers=workers,
            project=str(save_dir / 'models'),
            name=exp_name,
            exist_ok=True,
            pretrained=True,
            optimizer=config['train'].get('optimizer', 'AdamW'),
            weight_decay=config['train'].get('weight_decay', 0.0005),
            warmup_epochs=config['train'].get('warmup_epochs', 3),
            amp=config['train'].get('amp', True),
            mosaic=config['augmentation'].get('mosaic', 0.5),
            mixup=config['augmentation'].get('mixup', 0.2),
            copy_paste=config['augmentation'].get('copy_paste', 0.2),
            fliplr=config['augmentation'].get('fliplr', 0.5),
            translate=config['augmentation'].get('translate', 0.1),
            scale=config['augmentation'].get('scale', 0.5),
        )
        
        # 保存最佳模型
        best_model_path = save_dir / 'models' / exp_name / 'weights' / 'best.pt'
        if best_model_path.exists():
            print(f"\nBest model saved: {best_model_path}")
        
        # 记录实验结果
        tracker = ExperimentTracker(save_dir / 'logs')
        tracker.log_experiment(
            name='baseline_yolov10n',
            config=config,
            metrics={
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
            },
            model_path=str(best_model_path) if best_model_path.exists() else None
        )
        
        return results
        
    except ImportError:
        print("\n" + "!" * 60)
        print("ERROR: ultralytics not installed or YOLOv10 not available.")
        print("\nTo install YOLOv10:")
        print("  Option 1 (official):")
        print("    git clone https://github.com/THU-MIG/yolov10.git")
        print("    cd yolov10 && pip install -e .")
        print("  Option 2 (via ultralytics):")
        print("    pip install ultralytics")
        print("!" * 60)
        return None
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Baseline YOLOv10n Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, default=None, help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    
    args = parser.parse_args()
    
    # 更新配置（如果指定了命令行参数）
    if args.data:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['data']['dataset_yaml'] = args.data
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    results = train_baseline(args.config)
    
    if results is None:
        print("\nTraining failed. Running simulation mode...")
        run_simulation_mode()


def run_simulation_mode():
    """当真实训练不可用时，模拟训练流程并生成预期结果报告"""
    print("\n" + "=" * 60)
    print("Simulation Mode: Generating Expected Results")
    print("=" * 60)
    
    # 使用合成数据快速验证
    from data.prepare_dataset import create_synthetic_test_data
    create_synthetic_test_data('./data/processed', n_images=30)
    
    # 测试YOLO模型加载
    try:
        model = YOLO('yolov10n.pt')
        print("YOLOv10n model loaded successfully!")
        
        # 快速测试推理
        import cv2
        test_img = cv2.imread('./data/processed/images/test/synthetic_0000.jpg')
        if test_img is not None:
            results = model(test_img)
            print(f"Test inference completed: {len(results)} detections")
    except Exception as e:
        print(f"YOLO test: {e}")
        print("(Expected when no GPU/ultralytics available)")
    
    print("\n✓ Experiment 1 setup complete!")
    print("To run actual training:")
    print("  python train/train_baseline.py")


if __name__ == '__main__':
    main()
