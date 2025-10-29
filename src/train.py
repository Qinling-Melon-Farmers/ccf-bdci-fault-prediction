"""
模型训练脚本
实现完整的训练与评估流程，包括数据准备、模型训练、性能评估和结果可视化
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import DATASETS_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, WINDOW_SIZE
from build_dataset import build_train_dataset
from datasets import TimeSeriesNPZDataset, split_loaders
from models.cnn1d import CNN1D
from evaluate import evaluate_model


def train_one_run(train_loader, model, criterion, optimizer, device, epochs=EPOCHS):
    """
    执行一次完整的训练过程
    
    Args:
        train_loader: 训练数据加载器
        model: 待训练的模型
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epochs: 训练轮数
    """
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        epoch_losses = []
        
        # 遍历训练数据
        for x, y in train_loader:
            # 将数据移动到指定设备
            x = x.to(device)
            y = y.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            logits = model(x)
            # 计算损失
            loss = criterion(logits, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # 计算并打印平均损失
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")


def main():
    """主函数：执行完整的训练和评估流程"""
    # ==================== 数据准备 ====================
    train_npz = os.path.join(DATASETS_DIR, "train.npz")
    label_json = os.path.join(DATASETS_DIR, "label_map.json")
    
    # 如果数据集文件不存在，则构建数据集
    if not os.path.exists(train_npz) or not os.path.exists(label_json):
        print("正在构建数据集...")
        train_npz, label_json = build_train_dataset()

    # 加载数据集并划分训练/测试集
    dataset = TimeSeriesNPZDataset(train_npz)
    train_loader, test_loader = split_loaders(dataset, train_ratio=0.7, batch_size=BATCH_SIZE)

    # ==================== 设备配置 ====================
    # 自动检测CUDA是否可用并输出使用设备类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用的设备:", device)

    # ==================== 模型参数配置 ====================
    # 读取标签映射以确定类别数
    with open(label_json, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # 存储多次训练的结果
    acc_list, f1_list = [] , []

    # ==================== 多次训练实验 ====================
    # 按照赛题要求进行3次重复训练
    for run in range(1, 4):
        print(f"\n==== 第 {run} 次训练 ====")
        
        # 从数据集形状确定输入参数
        seq_len = dataset.samples.shape[1]  # 序列长度
        input_channels = dataset.samples.shape[2]  # 输入通道数

        # 创建模型并移动到指定设备
        model = CNN1D(input_channels=input_channels, num_classes=num_classes, seq_len=seq_len).to(device)
        
        # 使用交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 使用Adam优化器(lr=0.001)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 执行训练
        train_one_run(train_loader, model, criterion, optimizer, device, epochs=EPOCHS)
        
        # 评估模型性能
        acc, f1_macro, cm, test_labels, test_preds = evaluate_model(model, test_loader, device)
        acc_list.append(acc)
        f1_list.append(f1_macro)
        print(f"第{run}次评估结果: Accuracy={acc:.2f}%, F1-macro={f1_macro:.4f}")

        # ==================== 结果可视化 ====================
        # 绘制混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"混淆矩阵 - 第{run}次训练")
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        
        # 保存混淆矩阵图
        os.makedirs(MODELS_DIR, exist_ok=True)
        plt.savefig(os.path.join(MODELS_DIR, f"confusion_run_{run}.png"))
        plt.close()

        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"cnn1d_run_{run}.pt"))

    # ==================== 输出最终结果 ====================
    print("\n==== 三次训练结果平均值 ====")
    for i in range(3):
        print(f"Run {i+1}: Accuracy={acc_list[i]:.2f}%, F1-macro={f1_list[i]:.4f}")
    
    print(f"\n平均 Accuracy = {np.mean(acc_list):.2f}%")
    print(f"平均 F1-macro = {np.mean(f1_list):.4f}")


if __name__ == "__main__":
    main()