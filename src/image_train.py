"""
基于图像的信号分类训练脚本
将信号数据转换为图像，使用CNN进行分类
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# 导入自定义模块
from signal_to_image import SignalToImageConverter, SignalImageDataset
from models.image_cnn import create_model, count_parameters


class ImageTrainer:
    """图像分类训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 保存配置
        with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_data(self):
        """加载和预处理数据"""
        print("加载数据...")
        
        # 加载训练数据
        train_data = np.load(self.config['train_data_path'])
        X_train = train_data['samples']
        y_train = train_data['labels']
        
        print(f"训练数据形状: {X_train.shape}")
        print(f"标签形状: {y_train.shape}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        # 创建信号转图像转换器
        converter = SignalToImageConverter(
            image_size=self.config['image_size'],
            dpi=self.config['dpi'],
            style=self.config['image_style']
        )
        
        # 数据增强变换
        if self.config['use_augmentation']:
            train_transform = transforms.Compose([
                transforms.Resize(self.config['image_size']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建完整数据集
        full_dataset = SignalImageDataset(
            X_train, y_train, converter, 
            transform=train_transform,
            image_type=self.config['image_type']
        )
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return converter
    
    def create_model(self):
        """创建模型"""
        print(f"创建模型: {self.config['model_type']}")
        
        model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        )
        
        model = model.to(self.device)
        
        print(f"模型参数数量: {count_parameters(model):,}")
        
        return model
    
    def train_epoch(self, model, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, criterion):
        """验证模型"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_targets
    
    def train(self):
        """完整训练流程"""
        print("开始训练...")
        
        # 加载数据
        converter = self.load_data()
        
        # 创建模型
        model = self.create_model()
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(model, optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(model, criterion)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, best_model_path)
                print(f'  新的最佳模型已保存! 验证准确率: {val_acc:.2f}%')
        
        print(f'\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%')
        
        # 保存训练历史
        self.save_training_history(history)
        
        # 生成最终报告
        self.generate_final_report(model, val_preds, val_targets, best_val_acc)
        
        return model, history, best_val_acc
    
    def save_training_history(self, history):
        """保存训练历史"""
        # 保存数值数据
        np.save(os.path.join(self.config['output_dir'], 'training_history.npy'), history)
        
        # 绘制训练曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_curves.png'), dpi=300)
        plt.close()
    
    def generate_final_report(self, model, val_preds, val_targets, best_val_acc):
        """生成最终报告"""
        # 分类报告
        report = classification_report(val_targets, val_preds, output_dict=True)
        
        # 保存分类报告
        with open(os.path.join(self.config['output_dir'], 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 混淆矩阵
        cm = confusion_matrix(val_targets, val_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 生成文本报告
        report_text = f"""
图像分类训练报告
================

训练配置:
- 模型类型: {self.config['model_type']}
- 图像类型: {self.config['image_type']}
- 图像尺寸: {self.config['image_size']}
- 批次大小: {self.config['batch_size']}
- 学习率: {self.config['learning_rate']}
- 训练轮数: {self.config['epochs']}

训练结果:
- 最佳验证准确率: {best_val_acc:.2f}%
- 模型参数数量: {count_parameters(model):,}

分类报告:
{classification_report(val_targets, val_preds)}

训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(os.path.join(self.config['output_dir'], 'training_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("训练报告已保存!")


def main():
    parser = argparse.ArgumentParser(description='图像分类训练')
    parser.add_argument('--model_type', type=str, default='resnet18', 
                       choices=['custom', 'resnet18', 'resnet50', 'efficientnet_b0', 'attention'],
                       help='模型类型')
    parser.add_argument('--image_type', type=str, default='waveform',
                       choices=['waveform', 'spectrogram', 'combined'],
                       help='图像类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--output_dir', type=str, default='artifacts/image_training', help='输出目录')
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        'train_data_path': 'artifacts/datasets/train.npz',
        'model_type': args.model_type,
        'image_type': args.image_type,
        'image_size': (224, 224),
        'dpi': 100,
        'image_style': 'clean',
        'num_classes': 6,
        'pretrained': True,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'use_augmentation': True,
        'num_workers': 4,
        'output_dir': args.output_dir
    }
    
    # 创建训练器并开始训练
    trainer = ImageTrainer(config)
    model, history, best_acc = trainer.train()
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"结果保存在: {config['output_dir']}")


if __name__ == "__main__":
    main()