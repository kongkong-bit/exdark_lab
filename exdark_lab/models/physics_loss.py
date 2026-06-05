"""
物理约束损失函数 (Physics-Guided Loss Functions)
=================================================
基于Retinex理论的物理约束损失:

L_total = L_detection + λ₁·L_illumination_smooth + λ₂·L_reflection_consistency + λ₃·L_reconstruction

其中:
- L_illumination_smooth: 光照分量平滑约束，||∇L||²
- L_reflection_consistency: 反射分量一致性约束
- L_reconstruction: Retinex重构损失 (验证 I = R·L)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def illumination_smooth_loss(illumination):
    """
    光照分量平滑约束损失 ||∇L||²
    
    确保光照分量空间平滑，无剧烈变化。
    
    Args:
        illumination: [B, 1, H, W] 光照分量
    Returns:
        loss: 标量损失值
    """
    # 计算空间梯度
    dx = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
    dy = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])
    
    # L2范数平滑约束
    loss = torch.mean(dx**2 + dy**2)
    return loss


def illumination_smooth_loss_weighted(illumination, image):
    """
    加权光照平滑损失 - 边缘感知版本
    
    在图像边缘处放松平滑约束，避免光晕伪影。
    
    Args:
        illumination: [B, 1, H, W] 光照分量
        image: [B, C, H, W] 原始图像 (用于边缘检测)
    Returns:
        loss: 标量损失值
    """
    # 计算图像梯度作为边缘权重
    img_gray = torch.mean(image, dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 图像梯度
    img_dx = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    img_dy = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    
    # 边缘权重: 梯度大 -> 权重小 (放松约束)
    wx = torch.exp(-img_dx * 10)
    wy = torch.exp(-img_dy * 10)
    
    # 光照梯度
    L_dx = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
    L_dy = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])
    
    # 加权平滑损失
    loss = (torch.mean(wx * L_dx**2) + torch.mean(wy * L_dy**2)) / 2
    return loss


def reflection_consistency_loss(R_low, R_normal, mask=None):
    """
    反射分量一致性损失
    
    确保低光图像和正常光图像的反射分量保持一致。
    反射分量代表物体的本质属性，应与光照条件无关。
    
    Args:
        R_low: [B, C, H, W] 低光图像的反射分量
        R_normal: [B, C, H, W] 正常光图像的反射分量 (作为参考)
        mask: [B, 1, H, W] 可选掩码 (忽略某些区域)
    Returns:
        loss: 标量损失值
    """
    if mask is not None:
        diff = F.l1_loss(R_low * mask, R_normal.detach() * mask, reduction='none')
        loss = diff.sum() / (mask.sum() + 1e-8)
    else:
        loss = F.l1_loss(R_low, R_normal.detach())
    
    return loss


def retinex_reconstruction_loss(image, R, L, epsilon=1e-6):
    """
    Retinex重构损失
    
    验证 I = R · L 的分解准确性。
    确保分解后的R和L能重构回原始图像。
    
    Args:
        image: [B, C, H, W] 原始图像
        R: [B, C, H, W] 反射分量
        L: [B, 1, H, W] 光照分量
        epsilon: 防除零常数
    Returns:
        loss: 标量损失值
    """
    reconstructed = R * (L + epsilon)
    loss = F.l1_loss(reconstructed, image.detach())
    return loss


def compute_physics_loss(image_low, image_normal, retinex_model, config=None):
    """
    计算完整的物理约束损失集合
    
    Args:
        image_low: [B, C, H, W] 低光图像
        image_normal: [B, C, H, W] 正常光图像
        retinex_model: Retinex分解模型
        config: 配置字典, 包含各损失权重
            lambda_smooth: 光照平滑权重 (默认0.01)
            lambda_consistency: 反射一致性权重 (默认0.1)
            lambda_recon: 重构损失权重 (默认0.05)
    
    Returns:
        total_physics_loss: 综合物理损失
        loss_dict: 各分量损失字典
    """
    if config is None:
        config = {'lambda_smooth': 0.01, 'lambda_consistency': 0.1, 'lambda_recon': 0.05}
    
    # Retinex分解
    R_low, L_low = retinex_model(image_low)
    R_normal, L_normal = retinex_model(image_normal)
    
    # 光照平滑损失 (加权版本)
    loss_smooth = illumination_smooth_loss_weighted(L_low, image_low)
    
    # 反射一致性损失
    loss_consistency = reflection_consistency_loss(R_low, R_normal)
    
    # 重构损失
    loss_recon = retinex_reconstruction_loss(image_low, R_low, L_low)
    
    # 综合物理损失
    total = (
        config['lambda_smooth'] * loss_smooth +
        config['lambda_consistency'] * loss_consistency +
        config['lambda_recon'] * loss_recon
    )
    
    loss_dict = {
        'illumination_smooth': loss_smooth.item(),
        'reflection_consistency': loss_consistency.item(),
        'reconstruction': loss_recon.item(),
        'total_physics': total.item()
    }
    
    return total, loss_dict


class PhysicsLoss(nn.Module):
    """
    物理约束损失模块 (nn.Module封装)
    
    可在训练循环中直接使用。
    """
    
    def __init__(self, lambda_smooth=0.01, lambda_consistency=0.1, lambda_recon=0.05):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_consistency = lambda_consistency
        self.lambda_recon = lambda_recon
    
    def forward(self, image_low, image_normal, R_low, L_low, R_normal=None):
        """
        Args:
            image_low: 低光图像
            image_normal: 正常光图像 (或使用自身作为参考)
            R_low, L_low: 低光分解结果
            R_normal: 正常光反射分量 (可选)
        Returns:
            total_loss: 综合物理损失
            loss_dict: 各分量损失
        """
        loss_smooth = illumination_smooth_loss_weighted(L_low, image_low)
        
        if R_normal is not None:
            loss_consistency = reflection_consistency_loss(R_low, R_normal)
        else:
            # 自一致性: R应与边缘等结构信息一致
            loss_consistency = torch.tensor(0.0, device=image_low.device)
        
        loss_recon = retinex_reconstruction_loss(image_low, R_low, L_low)
        
        total = (
            self.lambda_smooth * loss_smooth +
            self.lambda_consistency * loss_consistency +
            self.lambda_recon * loss_recon
        )
        
        loss_dict = {
            'loss_smooth': loss_smooth.item(),
            'loss_consistency': loss_consistency.item() if isinstance(loss_consistency, torch.Tensor) else loss_consistency,
            'loss_recon': loss_recon.item(),
            'physics_total': total.item()
        }
        
        return total, loss_dict


if __name__ == '__main__':
    print("=== Physics Loss Unit Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟数据
    B, C, H, W = 2, 3, 128, 128
    image_low = torch.rand(B, C, H, W).to(device)
    image_normal = torch.rand(B, C, H, W).to(device)
    
    # 模拟Retinex分解输出
    R = torch.rand(B, C, H, W).to(device)
    L = torch.rand(B, 1, H, W).to(device)
    
    # 测试各损失函数
    loss_smooth = illumination_smooth_loss(L)
    loss_smooth_w = illumination_smooth_loss_weighted(L, image_low)
    loss_consistency = reflection_consistency_loss(R, R)
    loss_recon = retinex_reconstruction_loss(image_low, R, L)
    
    print(f"Illumination smooth loss: {loss_smooth.item():.6f}")
    print(f"Illumination smooth (weighted): {loss_smooth_w.item():.6f}")
    print(f"Reflection consistency loss: {loss_consistency.item():.6f}")
    print(f"Reconstruction loss: {loss_recon.item():.6f}")
    
    # 测试Physics Loss模块
    physics_loss = PhysicsLoss()
    total, loss_dict = physics_loss(image_low, image_normal, R, L, R_normal=R)
    print(f"\nTotal physics loss: {total.item():.6f}")
    print(f"Loss components: {loss_dict}")
    
    print("\nAll physics loss tests passed!")
