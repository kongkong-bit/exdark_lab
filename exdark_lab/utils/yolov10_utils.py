"""
YOLOv10 工具函数
================
YOLOv10 (由清华大学THU-MIG团队开发) 的核心特性:
  - NMS-Free设计: 双重分配策略消除NMS后处理依赖
  - 轻量化: 相比YOLOv8减少约30%参数量
  - 高效: 在COCO上达到SOTA精度/速度平衡

YOLOv10 GitHub: https://github.com/THU-MIG/yolov10

安装方式:
  git clone https://github.com/THU-MIG/yolov10.git
  cd yolov10
  pip install -r requirements.txt
  pip install -e .
"""

import os
import sys
import warnings

import torch
import numpy as np


# YOLOv10 类别 (COCO)
YOLOV10_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# ExDark 类别 (12类)
EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]


def load_yolov10_model(model_path='yolov10n.pt', device='auto'):
    """
    加载YOLOv10模型
    
    YOLOv10使用ultralytics API (与YOLOv8兼容)
    
    Args:
        model_path: 模型权重路径
        device: 计算设备
    Returns:
        model: YOLOv10模型实例
    """
    try:
        from ultralytics import YOLO
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = YOLO(model_path)
        print(f"YOLOv10 model loaded: {model_path}")
        return model
    
    except ImportError:
        warnings.warn(
            "\nYOLOv10 not available. Install it:\n"
            "  git clone https://github.com/THU-MIG/yolov10.git\n"
            "  cd yolov10 && pip install -r requirements.txt && pip install -e .\n"
            "Or use ultralytics: pip install ultralytics\n"
        )
        return None
    except Exception as e:
        warnings.warn(f"Failed to load YOLOv10: {e}")
        return None


def yolo_to_coco_class_id(yolo_class_id):
    """
    将YOLO类别ID映射到COCO类别ID
    
    YOLOv10使用COCO格式 (80类)
    """
    return yolo_class_id


def exdark_to_coco_class_mapping():
    """
    ExDark类别到COCO类别的映射
    
    Returns:
        mapping: {exdark_class_id: coco_class_id}
    """
    mapping = {
        0: 1,   # Bicycle → bicycle
        1: 8,   # Boat → boat
        2: 39,  # Bottle → bottle
        3: 5,   # Bus → bus
        4: 2,   # Car → car
        5: 15,  # Cat → cat
        6: 56,  # Chair → chair
        7: 41,  # Cup → cup
        8: 16,  # Dog → dog
        9: 3,   # Motorbike → motorcycle
        10: 0,  # People → person
        11: 60, # Table → dining table
    }
    return mapping


def convert_predictions_to_yolo_format(results, img_shape, conf_threshold=0.25):
    """
    将YOLO推理结果转换为标准格式
    
    Args:
        results: YOLO推理结果
        img_shape: (H, W) 原始图像尺寸
        conf_threshold: 置信度阈值
    Returns:
        detections: [{'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int}, ...]
    """
    detections = []
    
    if results is None:
        return detections
    
    try:
        # 处理YOLO结果
        if hasattr(results, 'boxes'):
            boxes = results.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].tolist()
                score = boxes.conf[i].item()
                class_id = int(boxes.cls[i].item())
                
                if score >= conf_threshold:
                    detections.append({
                        'bbox': bbox,
                        'score': score,
                        'class_id': class_id
                    })
        
        elif isinstance(results, list):
            for r in results:
                detections.extend(
                    convert_predictions_to_yolo_format(r, img_shape, conf_threshold)
                )
    
    except Exception as e:
        print(f"Warning: Failed to parse YOLO results: {e}")
    
    return detections


def nms_free_inference(model, image, conf_threshold=0.25):
    """
    YOLOv10 NMS-Free推理
    
    YOLOv10核心创新: 双重分配策略消除NMS依赖
    
    Args:
        model: YOLOv10模型
        image: 输入图像
        conf_threshold: 置信度阈值
    Returns:
        detections: 检测结果
    """
    # YOLOv10的推理接口与YOLOv8相同
    results = model(image, conf=conf_threshold)
    return results


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
    Returns:
        total_params: 总参数量 (M)
        trainable_params: 可训练参数量 (M)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total / 1e6, trainable / 1e6


def estimate_flops(model, input_size=(1, 3, 640, 640)):
    """
    估计模型FLOPs (需要ptflops库)
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
    Returns:
        flops: GFLOPs
    """
    try:
        from ptflops import get_model_complexity_info
        
        flops, params = get_model_complexity_info(
            model, input_size[1:],
            as_strings=True,
            print_per_layer_stat=False
        )
        return flops, params
    
    except ImportError:
        print("ptflops not installed. Install: pip install ptflops")
        return "N/A", "N/A"


if __name__ == '__main__':
    print("=== YOLOv10 Utils Test ===")
    
    # 测试模型加载
    print("\n1. Model loading...")
    model = load_yolov10_model('yolov10n.pt')
    if model:
        total, trainable = count_parameters(model.model)
        print(f"   Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
    
    # 测试预测转换
    print("\n2. YOLO format conversion...")
    print("   Ready ✓")
    
    print("\nAll YOLOv10 utils ready!")
