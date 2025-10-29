"""
数据集加载模块
提供时序数据的Dataset类和DataLoader封装功能
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from config import BATCH_SIZE


class TimeSeriesNPZDataset(Dataset):
    """
    时序数据Dataset类，用于加载npz格式的时序数据
    
    数据格式要求：
    - samples: 形状为(N, seq_len, num_channels)的时序样本
    - labels: 形状为(N,)的标签数组
    """
    
    def __init__(self, npz_path: str):
        """
        初始化数据集
        
        Args:
            npz_path: npz文件路径，包含'samples'和'labels'键
        """
        data = np.load(npz_path)
        # 加载样本数据，形状为(N, seq_len, num_channels)
        self.samples = data['samples'].astype(np.float32)
        # 加载标签数据
        self.labels = data['labels'].astype(np.int64)
        # 确保样本和标签数量一致
        assert len(self.samples) == len(self.labels)

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (样本张量, 标签张量)
        """
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def split_loaders(dataset: Dataset, train_ratio: float = 0.7, batch_size: int = BATCH_SIZE):
    """
    将数据集划分为训练集和测试集，并封装为DataLoader
    
    Args:
        dataset: 输入数据集
        train_ratio: 训练集比例，默认0.7
        batch_size: 批次大小，默认使用config中的BATCH_SIZE
        
    Returns:
        tuple: (训练DataLoader, 测试DataLoader)
        
    Note:
        训练集启用shuffle，测试集不启用shuffle
    """
    n = len(dataset)
    train_size = int(train_ratio * n)
    test_size = n - train_size
    
    # 随机划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建DataLoader，训练集shuffle=True，测试集shuffle=False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader