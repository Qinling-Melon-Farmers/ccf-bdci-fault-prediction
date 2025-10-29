"""
LSTM模型
基于长短期记忆网络的时序数据分类模型
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM分类模型
    
    网络结构：
    - 双向LSTM层
    - 注意力机制
    - 全连接分类层
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化LSTM模型
        
        Args:
            input_channels: 输入特征维度
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(LSTMModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = 128
        self.num_layers = 2
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,  # 双向LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch, seq_len, num_channels)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch, num_classes)
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output


class BiLSTMCNN(nn.Module):
    """
    BiLSTM + CNN混合模型
    结合LSTM的时序建模能力和CNN的局部特征提取能力
    """
    
    def __init__(self, input_channels: int, num_classes: int, seq_len: int = 500):
        """
        初始化BiLSTM-CNN模型
        
        Args:
            input_channels: 输入特征维度
            num_classes: 分类类别数
            seq_len: 输入序列长度，默认500
        """
        super(BiLSTMCNN, self).__init__()
        
        self.input_channels = input_channels
        self.lstm_hidden = 64
        
        # BiLSTM分支
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # CNN分支
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # 计算CNN输出维度
        self.cnn_output_size = self._get_cnn_output_size(seq_len)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2 + self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def _get_cnn_output_size(self, seq_len):
        """计算CNN分支的输出维度"""
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, seq_len)
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            return x.shape[1] * x.shape[2]
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch, seq_len, num_channels)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch, num_classes)
        """
        batch_size = x.size(0)
        
        # LSTM分支
        lstm_out, _ = self.lstm(x)
        lstm_pooled = torch.mean(lstm_out, dim=1)  # 全局平均池化
        
        # CNN分支
        x_cnn = x.transpose(1, 2)  # (batch, channels, seq_len)
        x_cnn = self.pool1(torch.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.pool2(torch.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(torch.relu(self.bn3(self.conv3(x_cnn))))
        cnn_pooled = x_cnn.reshape(batch_size, -1)
        
        # 特征融合
        fused = torch.cat([lstm_pooled, cnn_pooled], dim=1)
        
        # 分类
        output = self.fusion(fused)
        
        return output