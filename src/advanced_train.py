"""
高级训练脚本
整合数据增强、特征工程、超参数优化和模型集成
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, WINDOW_SIZE,
    NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    GENERATED_DIR, DATASETS_DIR, MODELS_DIR, SUBMISSION_DIR
)
from src.models import get_model
from src.data_augmentation import create_augmented_dataset
from src.feature_engineering import create_enhanced_features
from src.hyperparameter_optimization import HyperparameterOptimizer, create_hyperparameter_grid
from src.ensemble_predict import EnsemblePredictor


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, use_augmentation=True, use_enhanced_features=True, 
                 use_hyperopt=True, device='cuda'):
        """
        初始化高级训练器
        
        Args:
            use_augmentation: 是否使用数据增强
            use_enhanced_features: 是否使用增强特征
            use_hyperopt: 是否使用超参数优化
            device: 设备
        """
        self.use_augmentation = use_augmentation
        self.use_enhanced_features = use_enhanced_features
        self.use_hyperopt = use_hyperopt
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"使用设备: {self.device}")
        print(f"数据增强: {self.use_augmentation}")
        print(f"增强特征: {self.use_enhanced_features}")
        print(f"超参数优化: {self.use_hyperopt}")
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        if self.use_enhanced_features:
            # 使用增强特征
            try:
                train_enhanced = np.load(os.path.join(DATASETS_DIR, 'train_enhanced.npz'))
                test_enhanced = np.load(os.path.join(DATASETS_DIR, 'test_enhanced.npz'))
                
                X_train = train_enhanced['X']
                y_train = train_enhanced['y']
                X_test = test_enhanced['X']
                
                print(f"加载增强特征数据: {X_train.shape}")
            except FileNotFoundError:
                print("增强特征文件不存在，使用原始数据")
                self.use_enhanced_features = False
        
        if not self.use_enhanced_features:
            # 使用原始数据
            if self.use_augmentation:
                try:
                    train_data = np.load(os.path.join(DATASETS_DIR, 'train_augmented.npz'))
                    X_train = train_data['samples']
                    y_train = train_data['labels']
                    print(f"加载增强数据: {X_train.shape}")
                except FileNotFoundError:
                    print("增强数据文件不存在，使用原始数据")
                    train_data = np.load(os.path.join(DATASETS_DIR, 'train.npz'))
                    X_train = train_data['samples']
                    y_train = train_data['labels']
            else:
                train_data = np.load(os.path.join(DATASETS_DIR, 'train.npz'))
                X_train = train_data['samples']
                y_train = train_data['labels']
            
            test_data = np.load(os.path.join(DATASETS_DIR, 'test.npz'))
            X_test = test_data['samples']
        
        return X_train, y_train, X_test
    
    def prepare_data_loaders(self, X_train, y_train, test_size=0.2, batch_size=32):
        """准备数据加载器"""
        print("准备数据加载器...")
        
        # 划分训练集和验证集
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
        )
        
        # 转换为PyTorch张量
        if len(X_train_split.shape) == 2:
            # 增强特征是2D的，需要添加一个维度
            X_train_tensor = torch.FloatTensor(X_train_split).unsqueeze(1)  # (N, 1, features)
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        else:
            # 原始数据是3D的
            X_train_tensor = torch.FloatTensor(X_train_split)
            X_val_tensor = torch.FloatTensor(X_val)
        
        y_train_tensor = torch.LongTensor(y_train_split)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader, X_val, y_val
    
    def optimize_hyperparameters(self, model_name, train_loader, val_loader):
        """优化超参数"""
        if not self.use_hyperopt:
            return None
        
        print(f"\n开始优化 {model_name} 的超参数...")
        
        # 获取模型类
        model_class = get_model(model_name)
        
        # 创建超参数优化器
        optimizer = HyperparameterOptimizer(
            model_class, train_loader, val_loader, self.device
        )
        
        # 获取参数网格
        param_grids = create_hyperparameter_grid()
        
        if model_name.lower() in param_grids:
            param_grid = param_grids[model_name.lower()]
            
            # 进行网格搜索（限制试验次数以节省时间）
            best_params = optimizer.grid_search(
                param_grid, epochs=30, max_trials=20
            )
            
            # 保存结果
            results_path = os.path.join(MODELS_DIR, f'{model_name}_hyperopt_results.json')
            optimizer.save_results(results_path)
            
            return best_params
        else:
            print(f"未找到 {model_name} 的参数网格")
            return None
    
    def train_single_model(self, model_name, train_loader, val_loader, 
                          hyperparams=None, epochs=50):
        """训练单个模型"""
        print(f"\n训练 {model_name} 模型...")
        
        # 获取模型类
        model_class = get_model(model_name)
        
        # 根据数据维度调整模型参数
        sample_data = next(iter(train_loader))[0]
        if len(sample_data.shape) == 3 and sample_data.shape[1] == 1:
            # 增强特征数据 (N, 1, features)
            input_size = sample_data.shape[2]
            seq_length = 1
        else:
            # 原始数据 (N, seq_length, features)
            input_size = sample_data.shape[2]
            seq_length = sample_data.shape[1]
        
        # 获取模型
        if hyperparams and 'model_params' in hyperparams:
            model = model_class(**hyperparams['model_params'])
        else:
            # 使用默认参数，根据数据维度调整
            if model_name == 'CNN1D':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            elif model_name == 'ResNet1D':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            elif model_name == 'LSTMModel':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            elif model_name == 'BiLSTMCNN':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            elif model_name == 'TransformerModel':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            elif model_name == 'ConvTransformer':
                model = model_class(input_channels=input_size, num_classes=6, seq_len=seq_length)
            else:
                raise ValueError(f"未知模型: {model_name}")
        
        model = model.to(self.device)
        
        # 设置优化器和调度器
        if hyperparams:
            if hyperparams['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), 
                                           lr=hyperparams['lr'], 
                                           weight_decay=hyperparams['weight_decay'])
            elif hyperparams['optimizer'] == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), 
                                            lr=hyperparams['lr'], 
                                            weight_decay=hyperparams['weight_decay'])
            else:
                optimizer = torch.optim.SGD(model.parameters(), 
                                          lr=hyperparams['lr'], 
                                          momentum=0.9,
                                          weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        best_val_acc = 0.0
        patience = 0
        max_patience = 15
        
        for epoch in range(epochs):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
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
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 绘制训练历史
        try:
            self.plot_training_history(model_name, train_losses, train_accs, 
                                     val_losses, val_accs)
        except Exception as e:
            print(f"保存训练历史图时出错: {e}")
        
        return model, best_val_acc, {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
    
    def plot_training_history(self, model_name, train_losses, train_accs, 
                            val_losses, val_accs):
        """绘制训练历史"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 损失曲线
            ax1.plot(train_losses, label='训练损失')
            ax1.plot(val_losses, label='验证损失')
            ax1.set_title(f'{model_name} - 损失曲线')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # 准确率曲线
            ax2.plot(train_accs, label='训练准确率')
            ax2.plot(val_accs, label='验证准确率')
            ax2.set_title(f'{model_name} - 准确率曲线')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(GENERATED_DIR, f'{model_name}_training_history.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"训练历史图已保存到: {os.path.join(GENERATED_DIR, f'{model_name}_training_history.png')}")
            
        except ImportError:
            print("matplotlib未安装，跳过绘制训练历史图")
        except Exception as e:
            print(f"绘制训练历史图时出错: {e}")
    
    def train_all_models(self, train_loader, val_loader, X_val, y_val):
        """训练所有模型"""
        print("\n开始训练所有模型...")
        
        models_to_train = ['CNN1D', 'ResNet1D', 'LSTMModel', 'BiLSTMCNN', 
                          'TransformerModel', 'ConvTransformer']
        
        results = {}
        
        for model_name in models_to_train:
            try:
                # 超参数优化
                best_params = None
                if self.use_hyperopt:
                    best_params = self.optimize_hyperparameters(
                        model_name, train_loader, val_loader
                    )
                
                # 训练模型
                model, best_val_acc, history = self.train_single_model(
                    model_name, train_loader, val_loader, 
                    best_params, epochs=50
                )
                
                results[model_name] = {
                    'model': model,
                    'best_val_acc': best_val_acc,
                    'history': history,
                    'best_params': best_params
                }
                
                print(f"{model_name} 最佳验证准确率: {best_val_acc:.4f}")
                
            except Exception as e:
                print(f"训练 {model_name} 时出错: {e}")
                continue
        
        return results
    
    def ensemble_predict(self, results, X_test):
        """集成预测"""
        print("\n进行集成预测...")
        
        # 检查是否有成功训练的模型
        valid_models = {name: result for name, result in results.items() 
                       if result.get('model') is not None}
        
        if not valid_models:
            print("没有成功训练的模型，无法进行集成预测")
            return np.zeros(len(X_test), dtype=int)
        
        # 处理测试数据维度
        if len(X_test.shape) == 2:
            # 增强特征是2D的，需要添加一个维度
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # (N, 1, features)
        else:
            # 原始数据是3D的
            X_test_tensor = torch.FloatTensor(X_test)
        
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        
        for model_name, result in valid_models.items():
            model = result['model']
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0].to(self.device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    predictions.append(probs.cpu().numpy())
            
            predictions = np.vstack(predictions)
            all_predictions.append(predictions)
            print(f"{model_name} 预测完成")
        
        if len(all_predictions) == 0:
            print("没有有效的预测结果")
            return np.zeros(len(X_test), dtype=int)
        
        # 平均集成
        ensemble_probs = np.mean(all_predictions, axis=0)
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        return ensemble_predictions
    
    def run(self):
        """运行完整的训练流程"""
        print("开始高级训练流程...")
        
        # 1. 加载数据
        X_train, y_train, X_test = self.load_data()
        
        # 2. 准备数据加载器
        train_loader, val_loader, X_val, y_val = self.prepare_data_loaders(
            X_train, y_train, batch_size=32
        )
        
        # 3. 训练所有模型
        results = self.train_all_models(train_loader, val_loader, X_val, y_val)
        
        # 4. 输出结果总结
        print("\n=== 训练结果总结 ===")
        best_single_model = None
        best_single_acc = 0.0
        
        for model_name, result in results.items():
            acc = result['best_val_acc']
            print(f"{model_name}: {acc:.4f}")
            
            if acc > best_single_acc:
                best_single_acc = acc
                best_single_model = model_name
        
        print(f"\n最佳单模型: {best_single_model} ({best_single_acc:.4f})")
        
        # 5. 集成预测
        test_predictions = self.ensemble_predict(results, X_test)
        
        # 6. 保存预测结果
        submission_path = os.path.join(SUBMISSION_DIR, 'advanced_submission.csv')
        np.savetxt(submission_path, test_predictions, fmt='%d', delimiter=',')
        print(f"预测结果已保存到: {submission_path}")
        
        return results, test_predictions


if __name__ == "__main__":
    # 创建高级训练器
    trainer = AdvancedTrainer(
        use_augmentation=True,
        use_enhanced_features=True,
        use_hyperopt=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 运行训练
    results, predictions = trainer.run()
    
    print("\n高级训练完成!")