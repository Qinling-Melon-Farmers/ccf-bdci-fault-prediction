"""
图像分类CNN模型
用于对信号波形图像进行故障分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights


class WaveformCNN(nn.Module):
    """自定义波形图像CNN模型"""
    
    def __init__(self, num_classes=6, input_channels=3):
        super(WaveformCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetWaveform(nn.Module):
    """基于ResNet的波形图像分类模型"""
    
    def __init__(self, num_classes=6, model_name='resnet18', pretrained=True):
        super(ResNetWaveform, self).__init__()
        
        if model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 替换最后的分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class EfficientNetWaveform(nn.Module):
    """基于EfficientNet的波形图像分类模型"""
    
    def __init__(self, num_classes=6, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetWaveform, self).__init__()
        
        if model_name == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 替换最后的分类层
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class AttentionWaveformCNN(nn.Module):
    """带注意力机制的波形图像CNN模型"""
    
    def __init__(self, num_classes=6, input_channels=3):
        super(AttentionWaveformCNN, self).__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 注意力权重
        attention_weights = self.attention(features)
        
        # 应用注意力
        attended_features = features * attention_weights
        
        # 全局池化
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output


def create_model(model_type='resnet18', num_classes=6, pretrained=True):
    """
    创建模型的工厂函数
    
    Args:
        model_type: 模型类型 ('custom', 'resnet18', 'resnet50', 'efficientnet_b0', 'attention')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        
    Returns:
        PyTorch模型
    """
    if model_type == 'custom':
        return WaveformCNN(num_classes=num_classes)
    elif model_type in ['resnet18', 'resnet50']:
        return ResNetWaveform(num_classes=num_classes, model_name=model_type, pretrained=pretrained)
    elif model_type == 'efficientnet_b0':
        return EfficientNetWaveform(num_classes=num_classes, model_name=model_type, pretrained=pretrained)
    elif model_type == 'attention':
        return AttentionWaveformCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试不同模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = ['custom', 'resnet18', 'resnet50', 'efficientnet_b0', 'attention']
    
    for model_type in models_to_test:
        print(f"\n=== 测试 {model_type} 模型 ===")
        
        try:
            model = create_model(model_type, num_classes=6, pretrained=True)
            model = model.to(device)
            
            # 测试输入
            test_input = torch.randn(4, 3, 224, 224).to(device)
            
            # 前向传播
            with torch.no_grad():
                output = model(test_input)
            
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"参数数量: {count_parameters(model):,}")
            
        except Exception as e:
            print(f"模型 {model_type} 测试失败: {e}")
    
    print("\n模型测试完成！")