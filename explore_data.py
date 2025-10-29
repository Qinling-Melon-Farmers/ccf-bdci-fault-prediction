import numpy as np
import matplotlib.pyplot as plt

# 加载数据
train_data = np.load('artifacts/datasets/train.npz')
test_data = np.load('artifacts/datasets/test.npz')

print('=== 数据集基本信息 ===')
print(f'训练数据形状: {train_data["samples"].shape}')
print(f'训练标签形状: {train_data["labels"].shape}')
print(f'测试数据形状: {test_data["samples"].shape}')

print('\n=== 信号特征分析 ===')
X_train = train_data['samples']
y_train = train_data['labels']

print(f'时间步长: {X_train.shape[1]}')
print(f'信号通道数: {X_train.shape[2]}')
print(f'故障类型数量: {len(np.unique(y_train))}')
print(f'故障类型: {np.unique(y_train)}')

print('\n=== 各类别样本数量 ===')
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f'类别 {label}: {count} 个样本')

print('\n=== 信号数值范围 ===')
print(f'通道1 - 最小值: {X_train[:,:,0].min():.4f}, 最大值: {X_train[:,:,0].max():.4f}')
print(f'通道2 - 最小值: {X_train[:,:,1].min():.4f}, 最大值: {X_train[:,:,1].max():.4f}')

print('\n=== 样本信号统计 ===')
print(f'通道1 - 均值: {X_train[:,:,0].mean():.4f}, 标准差: {X_train[:,:,0].std():.4f}')
print(f'通道2 - 均值: {X_train[:,:,1].mean():.4f}, 标准差: {X_train[:,:,1].std():.4f}')