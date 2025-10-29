"""
一维残差神经网络模型
基于ResNet思想的1D卷积网络，用于时序数据分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D残差块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    一维残差神经网络
    
    网络结构：
    - 初始卷积层
    - 4个残差块组
    - 全局平均池化
    - 分类层
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化ResNet1D模型
        
        Args:
            input_channels: 输入通道数（特征维度）
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(ResNet1D, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差块组
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 分类层
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建残差块组"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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
        
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # 残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.fc(x)
        
        return x