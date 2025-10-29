"""
集成模型训练脚本
训练多个不同架构的模型用于后续集成
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.models import get_model


class SignalDataset(Dataset):
    """简单的信号数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_single_model(model, train_loader, val_loader, model_name, device, epochs=50):
    """
    训练单个模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_name: 模型名称
        device: 设备
        epochs: 训练轮数
        
    Returns:
        tuple: (最佳验证准确率, 训练历史)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    model.to(device)
    
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    print(f"\n开始训练 {model_name} 模型...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_preds.extend(pred.cpu().numpy())
            train_labels.extend(target.cpu().numpy())
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'{model_name}_best.pt'))
        
        print(f'Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}')
    
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    return best_val_acc, history


def plot_training_history(histories, model_names):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    axes[0, 0].set_title('训练损失')
    for i, (name, history) in enumerate(zip(model_names, histories)):
        axes[0, 0].plot(history['train_losses'], label=name)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练准确率
    axes[0, 1].set_title('训练准确率')
    for i, (name, history) in enumerate(zip(model_names, histories)):
        axes[0, 1].plot(history['train_accs'], label=name)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 验证准确率
    axes[1, 0].set_title('验证准确率')
    for i, (name, history) in enumerate(zip(model_names, histories)):
        axes[1, 0].plot(history['val_accs'], label=name)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 最佳验证准确率对比
    best_accs = [max(history['val_accs']) for history in histories]
    axes[1, 1].bar(model_names, best_accs)
    axes[1, 1].set_title('最佳验证准确率对比')
    axes[1, 1].set_ylabel('Best Validation Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 在柱状图上显示数值
    for i, acc in enumerate(best_accs):
        axes[1, 1].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'ensemble_training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型保存目录
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_data = np.load(TRAIN_DATA_PATH)
    X_train, y_train = train_data['X'], train_data['y']
    
    # 数据集划分
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 创建数据集和数据加载器
    train_dataset = SignalDataset(X_train_split, y_train_split)
    val_dataset = SignalDataset(X_val_split, y_val_split)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"输入维度: {X_train.shape}")
    print(f"类别数: {len(np.unique(y_train))}")
    
    # 定义要训练的模型
    model_configs = [
        ('cnn1d', 'CNN1D'),
        ('resnet1d', 'ResNet1D'),
        ('lstm', 'LSTM'),
        ('bilstm_cnn', 'BiLSTM-CNN'),
        ('transformer', 'Transformer'),
        ('conv_transformer', 'ConvTransformer')
    ]
    
    # 模型参数
    model_params = {
        'input_channels': X_train.shape[2],
        'num_classes': len(np.unique(y_train)),
        'seq_len': X_train.shape[1]
    }
    
    # 训练所有模型
    best_accs = []
    histories = []
    model_names = []
    
    for model_key, model_display_name in model_configs:
        try:
            print(f"\n{'='*50}")
            print(f"训练 {model_display_name} 模型")
            print(f"{'='*50}")
            
            # 创建模型
            model = get_model(model_key, **model_params)
            
            # 训练模型
            best_acc, history = train_single_model(
                model, train_loader, val_loader, 
                model_key, device, epochs=EPOCHS
            )
            
            best_accs.append(best_acc)
            histories.append(history)
            model_names.append(model_display_name)
            
            print(f"{model_display_name} 最佳验证准确率: {best_acc:.4f}")
            
        except Exception as e:
            print(f"训练 {model_display_name} 时出错: {str(e)}")
            continue
    
    # 绘制训练历史
    if histories:
        plot_training_history(histories, model_names)
    
    # 输出结果总结
    print(f"\n{'='*60}")
    print("集成训练结果总结")
    print(f"{'='*60}")
    
    for name, acc in zip(model_names, best_accs):
        print(f"{name:15}: {acc:.4f}")
    
    if best_accs:
        best_model_idx = np.argmax(best_accs)
        print(f"\n最佳单模型: {model_names[best_model_idx]} (准确率: {best_accs[best_model_idx]:.4f})")
        print(f"平均准确率: {np.mean(best_accs):.4f}")
        print(f"准确率标准差: {np.std(best_accs):.4f}")


if __name__ == "__main__":
    main()