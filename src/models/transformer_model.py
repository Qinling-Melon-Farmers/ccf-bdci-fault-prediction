"""
Transformer模型
基于自注意力机制的时序数据分类模型
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """
    Transformer分类模型
    
    网络结构：
    - 输入投影层
    - 位置编码
    - Transformer编码器
    - 分类头
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化Transformer模型
        
        Args:
            input_channels: 输入特征维度
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(TransformerModel, self).__init__()
        
        self.input_channels = input_channels
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 6
        
        # 输入投影
        self.input_projection = nn.Linear(input_channels, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch, seq_len, num_channels)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch, num_classes)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output


class ConvTransformer(nn.Module):
    """
    卷积-Transformer混合模型
    先用卷积提取局部特征，再用Transformer建模全局依赖
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化ConvTransformer模型
        
        Args:
            input_channels: 输入特征维度
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(ConvTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.d_model = 256
        
        # 卷积特征提取器
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU()
        )
        
        # 计算卷积后的序列长度
        self.conv_seq_len = seq_len // 4  # 两次池化，每次减半
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.conv_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch, seq_len, num_channels)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch, num_classes)
        """
        batch_size = x.size(0)
        
        # 卷积特征提取
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output