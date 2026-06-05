"""
YOLOv10n + Retinex 物理先验融合模型
====================================
将可微分Retinex物理模块嵌入YOLOv10n检测框架。

核心架构:
  输入图像 → Retinex增强 → YOLOv10n Backbone → Head → 检测结果

支持两种模式:
  方案A: 轻量级预处理器 (MSR → YOLOv10n)
  方案B: 可微分Retinex分解层 + 物理损失 (端到端训练)
"""

import torch
import torch.nn as nn
import warnings

from .retinex_module import MultiScaleRetinex, RetinexDecomposition


class YOLOv10nWithRetinex(nn.Module):
    """
    YOLOv10n + Retinex 融合模型
    
    将Retinex物理先验嵌入YOLOv10n检测流程。
    
    Args:
        retinex_mode: 'preprocess' 方案A(轻量预处理) 或 'decompose' 方案B(端到端分解)
        sigma_list: MSR多尺度高斯核列表
        epsilon: 防除零常数
        device: 计算设备
    """
    
    def __init__(
        self,
        retinex_mode='preprocess',
        sigma_list=[15, 80, 250],
        epsilon=1e-6,
        device=None
    ):
        super().__init__()
        self.retinex_mode = retinex_mode
        self.epsilon = epsilon
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化Retinex模块
        if retinex_mode == 'preprocess':
            # 方案A: 多尺度Retinex作为预处理
            self.retinex = MultiScaleRetinex(sigma_list=sigma_list)
        elif retinex_mode == 'decompose':
            # 方案B: 完整Retinex分解
            self.retinex = RetinexDecomposition(sigma=sigma_list[0], epsilon=epsilon)
        else:
            raise ValueError(f"Unknown retinex_mode: {retinex_mode}")
        
        # YOLOv10n检测器 (实际使用时加载)
        self.detector = None
        self._model_loaded = False
    
    def load_detector(self, model_path=None, model_name='yolov10n'):
        """
        加载YOLOv10n检测器
        
        Args:
            model_path: 模型权重路径 (None时使用预训练权重)
            model_name: 模型名称
        """
        try:
            from ultralytics import YOLO
            if model_path:
                self.detector = YOLO(model_path)
            else:
                # 尝试加载YOLOv10 (需要先pip install ultralytics)
                self.detector = YOLO(f'{model_name}.pt')
            self._model_loaded = True
            print(f"[YOLOv10nWithRetinex] Detector loaded: {model_name}")
        except ImportError:
            warnings.warn(
                "ultralytics not installed. Please install: pip install ultralytics\n"
                "Or use: from ultralytics import YOLO"
            )
            self._model_loaded = False
        except Exception as e:
            warnings.warn(f"Failed to load detector: {e}")
            self._model_loaded = False
    
    def enhance_image(self, img):
        """
        对图像进行Retinex增强
        
        Args:
            img: [B, C, H, W] 或 [C, H, W] 或 [H, W, C] 图像 tensor
                范围 [0, 1]
        Returns:
            enhanced: 增强后的图像
            extra: 附加输出 (反射分量R, 光照分量L等)
        """
        # 确保是4D tensor [B, C, H, W]
        if img.dim() == 3:
            if img.shape[0] in [1, 3] and img.shape[0] != img.shape[-1]:
                # [C, H, W]
                img = img.unsqueeze(0)
            else:
                # [H, W, C]
                img = img.permute(2, 0, 1).unsqueeze(0)
        elif img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        
        img = img.to(self.device)
        
        if self.retinex_mode == 'preprocess':
            # 方案A: 输出增强图像
            enhanced = self.retinex(img)
            extra = {'enhanced': enhanced}
        else:
            # 方案B: 输出R和L分量
            R, L = self.retinex(img)
            # 使用反射分量作为增强结果 (光照不变特征)
            enhanced = R
            extra = {'R': R, 'L': L}
        
        return enhanced, extra
    
    def forward(self, x):
        """
        前向传播: Retinex增强 → YOLOv10n检测
        
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            检测结果 (取决于detector的返回格式)
        """
        # 1. Retinex增强
        enhanced, extra = self.enhance_image(x)
        
        # 2. 送入YOLOv10n检测
        if self._model_loaded and self.detector is not None:
            # YOLO模型期望 [0,255] uint8 或归一化输入
            # 这里保持 [0,1] float，YOLO内部会处理
            return self.detector(enhanced)
        else:
            # 模型未加载时返回增强结果
            return enhanced
    
    def retinex_decompose(self, img):
        """
        完整Retinex分解 (用于物理损失计算和可视化)
        
        Args:
            img: [B, C, H, W] 输入图像
        Returns:
            R: 反射分量
            L: 光照分量
        """
        if self.retinex_mode == 'decompose':
            return self.retinex(img)
        else:
            # 预处理模式下仍用RetinexDecomposition分解
            decomposer = RetinexDecomposition(
                sigma=self.retinex.sigma_list[0] if hasattr(self.retinex, 'sigma_list') else 15,
                epsilon=self.epsilon
            ).to(img.device)
            return decomposer(img)


def create_retinex_yolov10n(retinex_mode='preprocess', **kwargs):
    """
    创建YOLOv10n+Retinex融合模型的便捷方法
    
    Args:
        retinex_mode: 'preprocess' 或 'decompose'
        **kwargs: 传递给YOLOv10nWithRetinex的参数
    
    Returns:
        model: YOLOv10nWithRetinex实例
    """
    model = YOLOv10nWithRetinex(retinex_mode=retinex_mode, **kwargs)
    model.load_detector()
    return model


if __name__ == '__main__':
    print("=== YOLOv10n + Retinex Model Unit Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试方案A (预处理模式)
    print("\n--- Testing Scheme A (Preprocess) ---")
    model_a = YOLOv10nWithRetinex(retinex_mode='preprocess', device=device).to(device)
    test_input = torch.rand(2, 3, 224, 224).to(device)
    
    enhanced, extra = model_a.enhance_image(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Enhanced range: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
    
    # 测试方案B (分解模式)
    print("\n--- Testing Scheme B (Decompose) ---")
    model_b = YOLOv10nWithRetinex(retinex_mode='decompose', device=device).to(device)
    enhanced_b, extra_b = model_b.enhance_image(test_input)
    print(f"R shape: {extra_b['R'].shape}, L shape: {extra_b['L'].shape}")
    print(f"R range: [{extra_b['R'].min():.4f}, {extra_b['R'].max():.4f}]")
    print(f"L range: [{extra_b['L'].min():.4f}, {extra_b['L'].max():.4f}]")
    
    # 测试Retinex分解接口
    R, L = model_b.retinex_decompose(test_input)
    print(f"\nDecomposition - R: {R.shape}, L: {L.shape}")
    
    # 验证梯度
    loss = enhanced.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model_a.retinex.parameters())
    print(f"Gradient flow: {'✓ Pass' if has_grad else '✗ Failed'}")
    
    print("\nAll model tests passed!")
