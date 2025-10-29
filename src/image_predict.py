"""
图像分类方法测试集预测脚本
使用训练好的图像分类模型对测试集进行预测
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from signal_to_image import SignalToImageConverter, SignalImageDataset
from models.image_cnn import create_model


class ImagePredictor:
    """图像分类预测器"""
    
    def __init__(self, model_dir='artifacts/image_training'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 创建信号转图像转换器
        self.converter = SignalToImageConverter(
            image_size=tuple(self.config['image_size']),
            dpi=self.config['dpi'],
            style=self.config['image_style']
        )
        
        # 加载模型
        self.model = self._load_model()
        
        print(f"图像预测器初始化完成")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.config['model_type']}")
        print(f"图像类型: {self.config['image_type']}")
    
    def _load_model(self):
        """加载训练好的模型"""
        # 创建模型
        model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            pretrained=False  # 加载已训练的权重
        )
        
        # 加载权重
        model_path = os.path.join(self.model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型权重: {model_path}")
            print(f"模型验证准确率: {checkpoint.get('val_acc', 'Unknown'):.4f}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def predict_test_set(self, test_data_path='artifacts/datasets/test.npz', 
                        batch_size=32, output_file='artifacts/image_submission.txt'):
        """对测试集进行预测"""
        print("开始加载测试数据...")
        
        # 加载测试数据
        test_data = np.load(test_data_path)
        test_samples = test_data['samples']  # (N, 500, 2)
        test_file_ids = test_data['file_ids']  # 文件ID
        
        print(f"测试集样本数量: {len(test_samples)}")
        print(f"信号形状: {test_samples.shape}")
        
        # 创建测试数据集
        # 为测试集创建虚拟标签
        dummy_labels = np.zeros(len(test_samples))
        
        test_dataset = SignalImageDataset(
            signal_data=test_samples,
            labels=dummy_labels,  # 测试集使用虚拟标签
            converter=self.converter,
            image_type=self.config['image_type'],
            transform=None  # 测试时不使用数据增强
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # 进行预测
        print("开始预测...")
        predictions = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="预测进度"):
                if isinstance(batch_data, (list, tuple)):
                    batch_images = batch_data[0]  # 只取图像，忽略标签
                else:
                    batch_images = batch_data
                
                batch_images = batch_images.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_images)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                predictions.extend(predicted_classes.cpu().numpy())
        
        # 生成提交文件
        self._generate_submission_file(test_file_ids, predictions, output_file)
        
        return predictions
    
    def _generate_submission_file(self, file_ids, predictions, output_file):
        """生成提交文件"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 创建提交数据
        submission_data = []
        for file_id, pred in zip(file_ids, predictions):
            submission_data.append({
                '测试集名称': file_id,
                '故障类型': int(pred)
            })
        
        # 保存为DataFrame
        df = pd.DataFrame(submission_data)
        
        # 保存为制表符分隔的文本文件
        df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
        
        print(f"提交文件已保存: {output_file}")
        print(f"预测结果统计:")
        print(df['故障类型'].value_counts().sort_index())
        
        return df
    
    def predict_single_signal(self, signal):
        """预测单个信号"""
        # 转换为图像
        image = self.converter.signal_to_image(signal, image_type=self.config['image_type'])
        
        # 转换为tensor
        if len(image.shape) == 3:  # (H, W, C)
            image = image.transpose(2, 0, 1)  # (C, H, W)
        image = torch.FloatTensor(image).unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        # 预测
        with torch.no_grad():
            output = self.model(image)
            probability = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probability, dim=1).item()
            confidence = probability[0, predicted_class].item()
        
        return predicted_class, confidence


def main():
    """主函数"""
    print("=== 图像分类方法测试集预测 ===")
    
    # 检查模型是否存在
    model_dir = 'artifacts/image_training'
    if not os.path.exists(os.path.join(model_dir, 'best_model.pth')):
        print("错误: 未找到训练好的模型文件!")
        print("请先运行 image_train.py 训练模型")
        return
    
    try:
        # 创建预测器
        predictor = ImagePredictor(model_dir)
        
        # 进行预测
        predictions = predictor.predict_test_set(
            test_data_path='artifacts/datasets/test.npz',
            batch_size=32,
            output_file='artifacts/image_submission.txt'
        )
        
        print(f"\n预测完成! 共预测 {len(predictions)} 个样本")
        print("提交文件已生成: artifacts/image_submission.txt")
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()