"""
一维卷积神经网络模型
用于时序数据分类的CNN1D模型实现
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    一维卷积神经网络模型
    
    网络结构：
    - 3个卷积块（卷积 + 批量归一化 + 最大池化）
    - 2个全连接层
    - 自动计算特征长度以避免硬编码
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化CNN1D模型
        
        Args:
            input_channels: 输入通道数（特征维度）
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(CNN1D, self).__init__()
        
        # 第一个卷积块：输入通道 -> 16通道
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        # 第二个卷积块：16通道 -> 32通道
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        # 第三个卷积块：32通道 -> 64通道
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        # 通过前向传播推断特征长度，避免硬编码
        self.feature_length = self._infer_feature_length(input_channels, seq_len)
        
        # 全连接层
        self.fc1 = nn.Linear(self.feature_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _infer_feature_length(self, input_channels: int, seq_len: int) -> int:
        """
        通过前向传播推断卷积层输出的特征长度
        
        Args:
            input_channels: 输入通道数
            seq_len: 输入序列长度
            
        Returns:
            int: 展平后的特征长度
        """
        with torch.no_grad():
            # 创建虚拟输入张量
            x = torch.zeros(1, input_channels, seq_len)
            # 通过卷积层计算输出形状
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            # 返回展平后的特征长度
            return x.shape[1] * x.shape[2]

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch, seq_len, num_channels)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch, num_classes)
        """
        # 转换维度：(batch, seq_len, num_channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        # 第二个卷积块
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        # 第三个卷积块
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # 展平特征
        x = x.reshape(x.size(0), -1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out