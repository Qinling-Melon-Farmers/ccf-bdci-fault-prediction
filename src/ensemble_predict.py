"""
集成预测脚本
使用多个训练好的模型进行预测融合
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

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


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, model_configs, model_params, device):
        """
        初始化集成预测器
        
        Args:
            model_configs: 模型配置列表
            model_params: 模型参数
            device: 设备
        """
        self.model_configs = model_configs
        self.model_params = model_params
        self.device = device
        self.models = {}
        self.model_weights = {}
        
    def load_models(self):
        """加载所有训练好的模型"""
        print("加载训练好的模型...")
        
        for model_key, model_name in self.model_configs:
            model_path = os.path.join(MODEL_SAVE_PATH, f'{model_key}_best.pt')
            
            if os.path.exists(model_path):
                try:
                    # 创建模型
                    model = get_model(model_key, **self.model_params)
                    
                    # 加载权重
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    
                    self.models[model_key] = model
                    print(f"✓ 成功加载 {model_name} 模型")
                    
                except Exception as e:
                    print(f"✗ 加载 {model_name} 模型失败: {str(e)}")
            else:
                print(f"✗ 未找到 {model_name} 模型文件: {model_path}")
        
        print(f"共加载了 {len(self.models)} 个模型")
    
    def calculate_model_weights(self, val_loader):
        """
        基于验证集性能计算模型权重
        
        Args:
            val_loader: 验证数据加载器
        """
        print("计算模型权重...")
        
        model_scores = {}
        
        for model_key, model in self.models.items():
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(target.cpu().numpy())
            
            accuracy = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='macro')
            
            # 使用F1分数作为权重基础
            model_scores[model_key] = f1
            print(f"{model_key}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # 归一化权重
        total_score = sum(model_scores.values())
        self.model_weights = {k: v / total_score for k, v in model_scores.items()}
        
        print("模型权重:")
        for model_key, weight in self.model_weights.items():
            print(f"  {model_key}: {weight:.4f}")
    
    def predict_proba(self, data_loader):
        """
        预测概率分布
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            numpy.ndarray: 预测概率
        """
        all_probs = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                batch_probs = []
                
                # 获取每个模型的预测
                for model_key, model in self.models.items():
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    
                    # 应用模型权重
                    weighted_probs = probs * self.model_weights[model_key]
                    batch_probs.append(weighted_probs.cpu().numpy())
                
                # 融合预测结果
                ensemble_probs = np.sum(batch_probs, axis=0)
                all_probs.append(ensemble_probs)
        
        return np.vstack(all_probs)
    
    def predict(self, data_loader):
        """
        预测类别
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            numpy.ndarray: 预测类别
        """
        probs = self.predict_proba(data_loader)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, data_loader):
        """
        评估集成模型性能
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            dict: 评估结果
        """
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                batch_probs = []
                
                # 获取每个模型的预测
                for model_key, model in self.models.items():
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    weighted_probs = probs * self.model_weights[model_key]
                    batch_probs.append(weighted_probs.cpu().numpy())
                
                # 融合预测结果
                ensemble_probs = np.sum(batch_probs, axis=0)
                preds = np.argmax(ensemble_probs, axis=1)
                
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'predictions': all_preds,
            'labels': all_labels
        }


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print("加载数据...")
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test = test_data['X']
    
    # 创建验证集用于计算模型权重
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 创建数据集和数据加载器
    val_dataset = SignalDataset(X_val, y_val)
    test_dataset = SignalDataset(X_test, np.zeros(len(X_test)))  # 测试集没有标签
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 模型配置
    model_configs = [
        ('cnn1d', 'CNN1D'),
        ('resnet1d', 'ResNet1D'),
        ('lstm', 'LSTM'),
        ('bilstm_cnn', 'BiLSTM-CNN'),
        ('transformer', 'Transformer'),
        ('conv_transformer', 'ConvTransformer')
    ]
    
    model_params = {
        'input_channels': X_train.shape[2],
        'num_classes': len(np.unique(y_train)),
        'seq_len': X_train.shape[1]
    }
    
    # 创建集成预测器
    ensemble = EnsemblePredictor(model_configs, model_params, device)
    
    # 加载模型
    ensemble.load_models()
    
    if len(ensemble.models) == 0:
        print("没有可用的模型，请先运行 ensemble_train.py 训练模型")
        return
    
    # 计算模型权重
    ensemble.calculate_model_weights(val_loader)
    
    # 在验证集上评估
    print("\n在验证集上评估集成模型...")
    val_results = ensemble.evaluate(val_loader)
    print(f"集成模型验证结果:")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  F1-macro: {val_results['f1_macro']:.4f}")
    
    # 对比单模型性能
    print("\n单模型 vs 集成模型对比:")
    print(f"{'模型':<15} {'Accuracy':<10} {'F1-macro':<10}")
    print("-" * 35)
    
    for model_key, model in ensemble.models.items():
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"{model_key:<15} {acc:<10.4f} {f1:<10.4f}")
    
    print(f"{'Ensemble':<15} {val_results['accuracy']:<10.4f} {val_results['f1_macro']:<10.4f}")
    
    # 生成测试集预测
    print("\n生成测试集预测...")
    test_predictions = ensemble.predict(test_loader)
    
    # 保存预测结果
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submit_ensemble.txt')
    
    with open(output_file, 'w') as f:
        for pred in test_predictions:
            f.write(f"{pred}\n")
    
    print(f"集成预测结果已保存到: {output_file}")
    print(f"测试样本数量: {len(test_predictions)}")
    
    # 统计预测分布
    unique, counts = np.unique(test_predictions, return_counts=True)
    print("\n预测类别分布:")
    for cls, count in zip(unique, counts):
        print(f"  类别 {cls}: {count} 个样本 ({count/len(test_predictions)*100:.1f}%)")


if __name__ == "__main__":
    main()