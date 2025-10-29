"""
模型包初始化文件
导入所有可用的模型类
"""

from .cnn1d import CNN1D
from .resnet1d import ResNet1D
from .lstm_model import LSTMModel, BiLSTMCNN
from .transformer_model import TransformerModel, ConvTransformer

__all__ = [
    'CNN1D',
    'ResNet1D', 
    'LSTMModel',
    'BiLSTMCNN',
    'TransformerModel',
    'ConvTransformer'
]

# 模型注册表，方便动态创建模型
MODEL_REGISTRY = {
    'CNN1D': CNN1D,
    'ResNet1D': ResNet1D,
    'LSTMModel': LSTMModel,
    'BiLSTMCNN': BiLSTMCNN,
    'TransformerModel': TransformerModel,
    'ConvTransformer': ConvTransformer,
    'cnn1d': CNN1D,
    'resnet1d': ResNet1D,
    'lstm': LSTMModel,
    'bilstm_cnn': BiLSTMCNN,
    'transformer': TransformerModel,
    'conv_transformer': ConvTransformer
}

def get_model(model_name: str):
    """
    根据模型名称获取模型类
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型类（未实例化）
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name]