"""
辅助函数库
==========
包含:
  1. 实验追踪 (ExperimentTracker)
  2. 随机种子设置
  3. 设备管理
  4. 日志工具
"""

import os
import sys
import random
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def setup_seed(seed=42):
    """
    设置所有随机种子，确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """
    获取可用设备 (GPU优先)
    
    Returns:
        device: torch.device or str 'cpu'
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = 'cpu'
        print("Using CPU (no GPU available)")
    
    return device


class ExperimentTracker:
    """
    实验追踪器
    
    记录和管理实验结果，支持:
    - 指标记录与保存
    - 实验结果对比
    - 自动生成报告
    """
    
    def __init__(self, log_dir):
        """
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.log_file = self.log_dir / f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        self._log("Experiment Tracker initialized")
    
    def _log(self, message):
        """记录日志到文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def log_experiment(self, name, config=None, metrics=None, model_path=None):
        """
        记录实验结果
        
        Args:
            name: 实验名称
            config: 实验配置
            metrics: 评估指标字典
            model_path: 模型保存路径
        """
        self.experiments[name] = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics or {},
            'model_path': model_path
        }
        
        self._log(f"Experiment '{name}' logged")
        if metrics:
            for k, v in metrics.items():
                self._log(f"  {k}: {v}")
    
    def save_results(self, filename='experiment_results.json'):
        """
        保存所有实验结果到JSON
        
        Args:
            filename: 文件名
        """
        save_path = self.log_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)
        
        self._log(f"Results saved: {save_path}")
        return save_path
    
    def get_comparison_table(self):
        """
        获取实验对比表格
        
        Returns:
            table_data: 表格数据 (列表格式)
        """
        if not self.experiments:
            return [["No experiments recorded"]]
        
        # 收集所有指标
        all_metrics = set()
        for exp in self.experiments.values():
            all_metrics.update(exp.get('metrics', {}).keys())
        all_metrics = sorted(all_metrics)
        
        # 表头
        table = [['Experiment'] + all_metrics]
        
        # 数据行
        for name, exp in self.experiments.items():
            row = [name]
            for metric in all_metrics:
                row.append(exp.get('metrics', {}).get(metric, 'N/A'))
            table.append(row)
        
        return table
    
    def print_summary(self):
        """打印实验摘要"""
        print("\n" + "=" * 60)
        print("Experiment Summary")
        print("=" * 60)
        
        for name, exp in self.experiments.items():
            print(f"\n[{name}]")
            if exp.get('metrics'):
                for k, v in exp['metrics'].items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        
        print("=" * 60)


def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
    Returns:
        formatted: 格式化字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
