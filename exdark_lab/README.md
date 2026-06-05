# AI4S物理辅助的低光目标识别系统

**Physics-Assisted Low-Light Object Detection with Retinex Prior + YOLOv10n + PINN Visualization**

---

## 项目简介

本项目基于 **Retinex物理先验**，将可微分的物理模型嵌入 **YOLOv10n** 检测框架，提升低光照条件下的目标检测性能。同时引入 **PINN风格可视化范式**，打开AI黑箱，直观展示物理先验如何提升模型鲁棒性与可解释性。

### 核心创新点

1. **架构创新**：将可微分Retinex物理模块嵌入YOLOv10n检测框架
2. **可视化创新**：引入4种PINN风格可视化范式
3. **量化验证**：通过ExDark数据集，物理先验带来实质性mAP提升

### 核心逻辑链

```
物理模型（Retinex分解）→ 提取光照不变特征（反射分量R）
→ 提升检测精度（mAP↑）→ PINN可视化验证（误差分布/Grad-CAM/分量解耦/损失地形）
```

---

## 环境配置

### 系统要求

- Python 3.10+
- 建议GPU: 8GB+显存 (RTX 3070/4060及以上)
- 无GPU时可在CPU上运行演示模式

### 安装依赖

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装YOLOv10 (官方库)
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install -r requirements.txt
pip install -e .
cd ..

# 3. 下载YOLOv10n预训练权重
# 自动下载: 运行训练脚本时会自动下载 yolov10n.pt
# 或手动下载: https://github.com/THU-MIG/yolov10/releases
```

### 依赖清单

核心依赖 (`requirements.txt`):

| 包 | 版本 | 用途 |
|---|---|---|
| torch | >=1.9 | 深度学习框架 |
| torchvision | >=0.10 | 图像处理 |
| ultralytics | >=8.0 | YOLO API |
| opencv-python | >=4.5 | 图像读写 |
| numpy | >=1.19 | 数值计算 |
| matplotlib | >=3.3 | 可视化 |
| scikit-image | >=0.18 | 图像指标 |
| pyyaml | >=5.4 | 配置文件 |
| tqdm | >=4.62 | 进度条 |
| tensorboard | >=2.7 | 训练可视化 |

---

## 数据集准备

### ExDark数据集 (推荐)

ExDark (Exclusively Dark Image Dataset) 是低光目标检测的权威基准:

- **7,363张** 低光照图片
- **12个类别**: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table
- **10种** 不同光照条件

```bash
# 下载ExDark (来自Kaggle或GitHub)
# https://www.kaggle.com/datasets/duttakolkata/exdark-dataset

# 运行预处理
python data/prepare_dataset.py --data_root ./data/ExDark --output_root ./data/processed
```

### 合成数据 (演示模式)

无真实数据集时，自动生成合成低光数据:

```bash
python data/prepare_dataset.py --synthetic --n_synthetic 100
```

### 数据集结构

```
data/processed/
├── images/
│   ├── train/    # ~5,890张 (80%)
│   ├── val/      # ~736张 (10%)
│   └── test/     # ~737张 (10%)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml  # YOLO配置文件
```

---

## 项目结构

```
0604/
├── data/
│   ├── ExDark/                 # 数据集 (需自行下载)
│   ├── prepare_dataset.py      # 预处理脚本
│   └── dataset.yaml            # YOLO配置文件
├── models/
│   ├── retinex_module.py       # 可微分Retinex分解模块
│   ├── physics_loss.py         # 物理约束损失
│   └── yolov10_retinex.py      # YOLOv10n+Retinex融合模型
├── train/
│   ├── train_baseline.py       # 实验1: 纯YOLOv10n
│   ├── train_enhanced.py       # 实验2: 传统增强预处理
│   └── train_physics.py        # 实验3: AI4S物理先验 (核心)
├── eval/
│   ├── compute_metrics.py      # mAP等指标计算
│   ├── visualize_error_dist.py # 范式一: 误差空间分布 ⭐
│   ├── visualize_gradcam.py    # 范式二: Grad-CAM热力图 ⭐
│   ├── visualize_decompose.py  # 范式三: Retinex分量解耦 ⭐
│   └── visualize_loss_landscape.py # 范式四: 损失地形 ⭐
├── utils/
│   ├── helpers.py              # 辅助函数
│   └── yolov10_utils.py        # YOLOv10工具函数
├── results/
│   ├── metrics/                # 指标CSV
│   ├── plots/                  # 可视化图表
│   ├── models/                 # 模型权重
│   └── logs/                   # 实验日志
├── config.yaml                 # 全局配置
├── requirements.txt            # 依赖清单
├── run_all_experiments.py      # 一键运行所有实验
└── README.md                   # 本文档
```

---

## 快速开始

### 一键运行所有实验 (演示模式)

```bash
python run_all_experiments.py
```

### 分别运行各实验

```bash
# 实验1: Baseline
python train/train_baseline.py

# 实验2: 传统增强 (CLAHE)
python train/train_enhanced.py --method clahe

# 实验3: AI4S物理先验 ⭐ (核心)
python train/train_physics.py
```

### 运行可视化

```bash
# 范式一: 误差空间分布
python eval/visualize_error_dist.py

# 范式二: Grad-CAM热力图
python eval/visualize_gradcam.py

# 范式三: Retinex分量解耦
python eval/visualize_decompose.py

# 范式四: 损失地形
python eval/visualize_loss_landscape.py

# 指标计算
python eval/compute_metrics.py

# 一键所有可视化
python run_all_experiments.py --visualize-only
```

### 运行为真实训练模式

```bash
# 需要GPU + 完整ExDark数据集
python run_all_experiments.py --real
```

---

## 三组实验设计

### 实验1: Baseline (纯数据驱动)

- 直接使用YOLOv10n在原始低光图像上训练
- **预期mAP@0.5**: ~78-80%

### 实验2: 传统增强预处理

- 使用CLAHE/直方图均衡化等方法增强后训练
- **预期mAP@0.5**: ~79-81%

### 实验3: AI4S物理先验 (核心贡献) ⭐

- 可微分Retinex物理模块 + YOLOv10n端到端训练
- 物理约束损失引导学习
- **预期mAP@0.5**: >82%

---

## PINN风格可视化 (核心创新) ⭐

### 范式一: 误差空间分布

**揭示"哪里预测不准"**

- Baseline: 误差集中在低光照区域 (红色大)
- Ours: 整个图像均匀且极小
- **结论**: 物理先验打破光照与检测误差的虚假相关性

### 范式二: Grad-CAM热力图

**展示"注意力纠偏"**

- Baseline: 热力图大面积激活在噪声/光晕上
- Ours: 热力图精准包裹目标轮廓
- **结论**: 物理先验使模型聚焦于正确的物理语义

### 范式三: Retinex分量解耦

**验证"光照不变性"**

- 反射分量R: 不同光照下保持清晰边缘
- 光照分量L: 平滑反映环境光照变化
- **结论**: R可作为光照不变的鲁棒特征

### 范式四: 损失地形 (可选)

**解释"优化过程"**

- Baseline: 平坦地形 → 易陷入局部最优
- Ours: 漏斗地形 → 快速收敛到物理合理解
- **结论**: 物理先验改变损失地形平坦度

---

## 预期实验结果

| 实验方案 | mAP@0.5 | 参数量 | GFLOPs | 反射一致性 |
|---------|---------|--------|--------|-----------|
| Baseline YOLOv10n | 78-80% | 2.7M | 6.7 | - |
| 传统增强预处理 | 79-81% | 2.7M | 6.7 | - |
| **AI4S物理先验 (Ours)** | **>82%** | 2.9M | 7.0 | <0.01 |

### 生成的可视化文件

```
results/plots/
├── error_distribution_comparison.png  # 范式一: 误差分布
├── gradcam_comparison.png             # 范式二: Grad-CAM
├── retinex_decomposition.png          # 范式三: 分量解耦
├── loss_landscape.png                 # 范式四: 损失地形
├── mAP_curves.png                     # mAP收敛曲线
├── enhancement_comparison.png         # 增强对比
└── confusion_matrix.png               # 混淆矩阵
```

---

## 算法细节

### Retinex理论

```
I(x,y) = R(x,y) · L(x,y)
```

- **I**: 观测图像
- **R**: 反射分量 (物体本质属性, 光照不变)
- **L**: 光照分量 (环境光照, 空间平滑)

### 实现方案

**方案A (轻量预处理器)**: 多尺度Retinex (MSR) 作为预处理

```python
class MultiScaleRetinex(nn.Module):
    # 多尺度高斯滤波 + log域差分 → sigmoid输出
```

**方案B (端到端分解)**: 可微分Retinex分解 + 物理损失

```python
L_total = L_detection + λ₁·L_illumination_smooth + λ₂·L_reflection_consistency
```

### YOLOv10n核心特性

- **NMS-Free**: 双重分配策略消除NMS后处理
- **轻量化**: 相比YOLOv8减少30%参数量
- **高效**: COCO上SOTA精度/速度平衡

---

## 结果解读

### 性能提升机制

1. **Retinex增强** → 提升低光图像可见度 → 检测器获得更清晰的输入
2. **物理约束** → 引导网络学习光照不变特征 → 减少光照变化的影响
3. **反射分量** → 作为检测器输入 → 打破噪声与目标的虚假相关性

### 可视化科学结论

- 物理先验成功引导网络提取**光照无关的本质特征**
- 打破了天气噪声与目标之间的**虚假相关性**
- 模型决策依据从"纹理伪影"回归到正确的**物理语义**

---

## 常见问题

### Q: 没有GPU能运行吗？

可以运行演示模式 (`--simulate`，默认)，所有可视化模块可在CPU上执行。
训练模式需要GPU。

### Q: 如何获取ExDark数据集？

- Kaggle: https://www.kaggle.com/datasets/duttakolkata/exdark-dataset
- GitHub: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

### Q: YOLOv10安装失败？

```bash
# 确保git已安装，然后:
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip install -r requirements.txt
pip install -e .
# 如果仍有问题，YOLOv10的API与ultralytics兼容:
pip install ultralytics
```

### Q: 显存不足怎么办？

- 使用 `yolov10n` (最轻量版本)
- 降低 `batch_size` 至 8 或 4
- 降低 `img_size` 至 416

---

## 引用

```bibtex
@article{retinex_yolov10_2024,
  title={Physics-Assisted Low-Light Object Detection via Differentiable Retinex Prior and PINN Visualization},
  author={AI4S Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 参考论文

- YOLOv10: `THU-MIG/yolov10` - NMS-free detection
- Retinex Theory: Land (1986) - Color vision model
- FE-YOLO: Fourier enhancement for low-light detection
- YOLO-SCEB: Lightweight detection on ExDark
- CoPINN (ICML 2025): Error space visualization
- VI-PINN: Image-to-physics parameter mapping

---

**物理先验 + 深度学习 = 更鲁棒、更可解释的AI系统 ⭐**
