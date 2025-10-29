"""
简化训练脚本
专门处理增强特征数据的训练
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR
)


class SimpleMLPModel(nn.Module):
    """简单的多层感知机模型，适用于增强特征数据"""
    
    def __init__(self, input_size, num_classes=6):
        super(SimpleMLPModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class SimpleTrainer:
    """简化的训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def load_data(self):
        """加载增强特征数据"""
        print("加载增强特征数据...")
        
        # 加载训练数据
        train_data = np.load(os.path.join(DATASETS_DIR, 'train_enhanced.npz'))
        X_train = train_data['X']
        y_train = train_data['y']
        
        # 加载测试数据
        test_data = np.load(os.path.join(DATASETS_DIR, 'test_enhanced.npz'))
        X_test = test_data['X']
        
        print(f"训练数据形状: {X_train.shape}")
        print(f"测试数据形状: {X_test.shape}")
        print(f"类别分布: {np.bincount(y_train)}")
        
        return X_train, y_train, X_test
    
    def prepare_data_loaders(self, X_train, y_train):
        """准备数据加载器"""
        # 划分训练集和验证集
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_split)
        y_train_tensor = torch.LongTensor(y_train_split)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, X_val, y_val
    
    def train_model(self, model, train_loader, val_loader, X_val, y_val):
        """训练模型"""
        model = model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_val_acc = 0.0
        best_model = None
        patience = 10
        patience_counter = 0
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        print("开始训练...")
        
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # 计算准确率
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}: "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
        
        print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        
        return model, best_val_acc
    
    def predict(self, model, X_test):
        """预测测试数据"""
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                data = batch[0].to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def save_submission(self, predictions, filename='simple_submission.txt'):
        """保存提交文件"""
        # 标签映射：数字标签转换为文本标签
        label_map = {
            0: 'inner_broken',
            1: 'inner_wear',
            2: 'normal',  # normal_train160 -> normal
            3: 'outer_missing',
            4: 'roller_broken',
            5: 'roller_wear'
        }
        
        # 加载测试数据的文件ID信息
        test_data = np.load(os.path.join(DATASETS_DIR, 'test.npz'))
        file_ids = test_data['file_ids']
        
        # 聚合每个文件的预测结果
        unique_file_ids = np.unique(file_ids)
        aggregated_predictions = {}
        
        print(f"聚合 {len(unique_file_ids)} 个测试文件的预测结果...")
        
        for file_id in unique_file_ids:
            # 找到该文件对应的所有样本索引
            indices = np.where(file_ids == file_id)[0]
            file_predictions = predictions[indices]
            
            # 使用投票机制聚合预测结果（选择出现次数最多的类别）
            unique_preds, counts = np.unique(file_predictions, return_counts=True)
            final_prediction = unique_preds[np.argmax(counts)]
            
            aggregated_predictions[file_id] = final_prediction
        
        submission_path = os.path.join(ARTIFACTS_DIR, filename)
        
        # 写入txt文件，使用制表符分隔
        with open(submission_path, 'w', encoding='utf-8') as f:
            # 写入标题行
            f.write('测试集名称\t故障类型\n')
            
            # 按文件ID排序写入预测结果
            for file_id in sorted(unique_file_ids):
                pred = aggregated_predictions[file_id]
                fault_type = label_map[pred]
                f.write(f'{file_id}\t{fault_type}\n')
        
        print(f"提交文件已保存到: {submission_path}")
        print(f"总共 {len(unique_file_ids)} 个测试文件的预测结果")
        
        return submission_path
    
    def run(self):
        """运行完整的训练流程"""
        # 加载数据
        X_train, y_train, X_test = self.load_data()
        
        # 准备数据加载器
        train_loader, val_loader, X_val, y_val = self.prepare_data_loaders(X_train, y_train)
        
        # 创建模型
        input_size = X_train.shape[1]
        model = SimpleMLPModel(input_size, NUM_CLASSES)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        trained_model, best_val_acc = self.train_model(model, train_loader, val_loader, X_val, y_val)
        
        # 保存模型
        model_path = os.path.join(MODELS_DIR, 'simple_mlp_model.pth')
        torch.save(trained_model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
        
        # 预测测试数据
        print("预测测试数据...")
        predictions = self.predict(trained_model, X_test)
        
        # 保存提交文件
        submission_path = self.save_submission(predictions)
        
        return {
            'model': trained_model,
            'best_val_acc': best_val_acc,
            'predictions': predictions,
            'submission_path': submission_path
        }


if __name__ == "__main__":
    # 创建训练器
    trainer = SimpleTrainer()
    
    # 运行训练
    results = trainer.run()
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {results['best_val_acc']:.2f}%")
    print(f"提交文件: {results['submission_path']}")