"""
模型评估模块
提供模型性能评估功能，包括准确率、宏F1分数和混淆矩阵计算
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


def evaluate_model(model, loader, device):
    """
    评估模型性能
    
    Args:
        model: 待评估的PyTorch模型
        loader: 数据加载器（DataLoader）
        device: 计算设备（CPU或GPU）
        
    Returns:
        tuple: (准确率, 宏F1分数, 混淆矩阵, 真实标签, 预测标签)
        
    Note:
        - 准确率以百分比形式返回
        - 使用宏平均F1分数，符合赛题要求
        - 在评估过程中禁用梯度计算以节省内存
    """
    model.eval()  # 设置模型为评估模式
    all_labels, all_preds = [], []
    
    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        for x, y in loader:
            # 将数据移动到指定设备
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            logits = model(x)
            # 获取预测类别
            preds = torch.argmax(logits, dim=1)
            
            # 收集真实标签和预测结果
            all_labels.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算评估指标
    accuracy = (all_labels == all_preds).mean() * 100  # 准确率（百分比）
    f1_macro = f1_score(all_labels, all_preds, average="macro")  # 宏平均F1分数
    cm = confusion_matrix(all_labels, all_preds)  # 混淆矩阵
    
    return accuracy, f1_macro, cm, all_labels, all_preds