"""
超参数优化模块
实现学习率调度、正则化和模型架构参数的优化
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)
from sklearn.model_selection import ParameterGrid
import itertools
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, model_class, train_loader, val_loader, device='cuda'):
        """
        初始化超参数优化器
        
        Args:
            model_class: 模型类
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
        """
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.best_params = None
        self.best_score = 0.0
        self.optimization_history = []
    
    def get_optimizer(self, model, optimizer_name, lr, weight_decay=0):
        """
        获取优化器
        
        Args:
            model: 模型
            optimizer_name: 优化器名称
            lr: 学习率
            weight_decay: 权重衰减
            
        Returns:
            torch.optim.Optimizer: 优化器
        """
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def get_scheduler(self, optimizer, scheduler_name, **kwargs):
        """
        获取学习率调度器
        
        Args:
            optimizer: 优化器
            scheduler_name: 调度器名称
            **kwargs: 调度器参数
            
        Returns:
            torch.optim.lr_scheduler: 学习率调度器
        """
        if scheduler_name == 'step':
            return StepLR(optimizer, step_size=kwargs.get('step_size', 10), 
                         gamma=kwargs.get('gamma', 0.1))
        elif scheduler_name == 'exponential':
            return ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.95))
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='max', factor=kwargs.get('factor', 0.5),
                                   patience=kwargs.get('patience', 5))
        elif scheduler_name == 'cyclic':
            return CyclicLR(optimizer, base_lr=kwargs.get('base_lr', 1e-5),
                           max_lr=kwargs.get('max_lr', 1e-2))
        elif scheduler_name == 'onecycle':
            return OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 1e-2),
                             steps_per_epoch=len(self.train_loader),
                             epochs=kwargs.get('epochs', 50))
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")
    
    def get_loss_function(self, loss_name, **kwargs):
        """
        获取损失函数
        
        Args:
            loss_name: 损失函数名称
            **kwargs: 损失函数参数
            
        Returns:
            torch.nn.Module: 损失函数
        """
        if loss_name == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            return FocalLoss(alpha=kwargs.get('alpha', 1.0), 
                           gamma=kwargs.get('gamma', 2.0))
        elif loss_name == 'label_smoothing':
            return LabelSmoothingLoss(smoothing=kwargs.get('smoothing', 0.1))
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def train_and_evaluate(self, params, epochs=50, verbose=False):
        """
        训练和评估模型
        
        Args:
            params: 超参数字典
            epochs: 训练轮次
            verbose: 是否打印详细信息
            
        Returns:
            float: 验证准确率
        """
        # 创建模型
        model = self.model_class(**params['model_params']).to(self.device)
        
        # 创建优化器
        optimizer = self.get_optimizer(
            model, params['optimizer'], params['lr'], params['weight_decay']
        )
        
        # 创建学习率调度器
        scheduler = self.get_scheduler(
            optimizer, params['scheduler'], **params.get('scheduler_params', {})
        )
        
        # 创建损失函数
        criterion = self.get_loss_function(
            params['loss'], **params.get('loss_params', {})
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = params.get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                if params.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
                
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
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"早停在第 {epoch+1} 轮")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        return best_val_acc
    
    def grid_search(self, param_grid, epochs=50, max_trials=None):
        """
        网格搜索超参数
        
        Args:
            param_grid: 参数网格
            epochs: 训练轮次
            max_trials: 最大试验次数
            
        Returns:
            dict: 最佳参数
        """
        print("开始网格搜索超参数优化...")
        
        # 生成所有参数组合
        param_combinations = list(ParameterGrid(param_grid))
        
        if max_trials and len(param_combinations) > max_trials:
            # 随机采样
            import random
            param_combinations = random.sample(param_combinations, max_trials)
        
        print(f"总共需要测试 {len(param_combinations)} 个参数组合")
        
        for i, params in enumerate(param_combinations):
            print(f"\n测试参数组合 {i+1}/{len(param_combinations)}")
            print(f"参数: {params}")
            
            try:
                score = self.train_and_evaluate(params, epochs, verbose=False)
                
                # 记录结果
                result = {
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                }
                self.optimization_history.append(result)
                
                print(f"验证准确率: {score:.4f}")
                
                # 更新最佳参数
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    print(f"发现更好的参数组合! 最佳得分: {self.best_score:.4f}")
                
            except Exception as e:
                print(f"参数组合测试失败: {e}")
                continue
        
        print(f"\n超参数优化完成!")
        print(f"最佳验证准确率: {self.best_score:.4f}")
        print(f"最佳参数: {self.best_params}")
        
        return self.best_params
    
    def save_results(self, filepath):
        """保存优化结果"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"优化结果已保存到: {filepath}")


class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_prob = torch.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def create_hyperparameter_grid():
    """创建超参数网格"""
    
    # CNN1D参数网格
    cnn1d_grid = {
        'model_params': [
            {'input_channels': 2, 'num_classes': 6, 'seq_length': 500}
        ],
        'optimizer': ['adam', 'adamw', 'sgd'],
        'lr': [0.001, 0.003, 0.01, 0.03],
        'weight_decay': [0, 1e-4, 1e-3, 1e-2],
        'scheduler': ['step', 'cosine', 'plateau', 'onecycle'],
        'scheduler_params': [
            {'step_size': 15, 'gamma': 0.1},
            {'T_max': 50},
            {'patience': 5, 'factor': 0.5},
            {'max_lr': 0.1, 'epochs': 50}
        ],
        'loss': ['crossentropy', 'focal', 'label_smoothing'],
        'loss_params': [
            {},
            {'alpha': 1.0, 'gamma': 2.0},
            {'smoothing': 0.1}
        ],
        'grad_clip': [0, 1.0, 5.0],
        'early_stopping_patience': [10, 15, 20]
    }
    
    # ResNet1D参数网格
    resnet1d_grid = {
        'model_params': [
            {'input_channels': 2, 'num_classes': 6, 'seq_length': 500, 'base_channels': 32},
            {'input_channels': 2, 'num_classes': 6, 'seq_length': 500, 'base_channels': 64},
            {'input_channels': 2, 'num_classes': 6, 'seq_length': 500, 'base_channels': 128}
        ],
        'optimizer': ['adam', 'adamw'],
        'lr': [0.001, 0.003, 0.01],
        'weight_decay': [1e-4, 1e-3],
        'scheduler': ['cosine', 'onecycle'],
        'scheduler_params': [
            {'T_max': 50},
            {'max_lr': 0.05, 'epochs': 50}
        ],
        'loss': ['crossentropy', 'focal'],
        'loss_params': [
            {},
            {'alpha': 1.0, 'gamma': 2.0}
        ],
        'grad_clip': [1.0, 5.0],
        'early_stopping_patience': [10, 15]
    }
    
    # LSTM参数网格
    lstm_grid = {
        'model_params': [
            {'input_size': 2, 'hidden_size': 64, 'num_layers': 2, 'num_classes': 6},
            {'input_size': 2, 'hidden_size': 128, 'num_layers': 2, 'num_classes': 6},
            {'input_size': 2, 'hidden_size': 256, 'num_layers': 3, 'num_classes': 6}
        ],
        'optimizer': ['adam', 'adamw'],
        'lr': [0.001, 0.003],
        'weight_decay': [1e-4, 1e-3],
        'scheduler': ['plateau', 'cosine'],
        'scheduler_params': [
            {'patience': 5, 'factor': 0.5},
            {'T_max': 50}
        ],
        'loss': ['crossentropy'],
        'loss_params': [{}],
        'grad_clip': [1.0, 5.0],
        'early_stopping_patience': [15, 20]
    }
    
    return {
        'cnn1d': cnn1d_grid,
        'resnet1d': resnet1d_grid,
        'lstm': lstm_grid
    }


if __name__ == "__main__":
    print("超参数优化模块已创建")
    print("使用示例:")
    print("1. 创建HyperparameterOptimizer实例")
    print("2. 调用grid_search方法进行网格搜索")
    print("3. 使用save_results保存优化结果")