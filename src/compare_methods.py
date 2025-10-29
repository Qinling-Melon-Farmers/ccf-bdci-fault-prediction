"""
方法对比脚本
比较时序模型和图像分类方法的性能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class MethodComparator:
    """方法对比器"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
    
    def load_time_series_results(self):
        """加载时序模型结果"""
        # 从simple_train.py的输出中提取结果
        # 这里我们使用已知的结果，实际应用中可以从日志文件中解析
        
        time_series_results = {
            'method': 'Time Series (1D CNN)',
            'model_type': '1D CNN',
            'input_type': 'Raw Signal',
            'input_shape': '(500, 2)',
            'accuracy': 36.34,  # 从之前的训练结果
            'parameters': 'Unknown',  # 需要从模型中获取
            'training_time': 'Unknown',
            'inference_speed': 'Fast',
            'memory_usage': 'Low',
            'interpretability': 'Medium'
        }
        
        self.results['time_series'] = time_series_results
        return time_series_results
    
    def load_image_results(self, result_dir='artifacts/image_training'):
        """加载图像分类结果"""
        image_results = {}
        
        if os.path.exists(result_dir):
            # 加载配置
            config_path = os.path.join(result_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 加载训练历史
                history_path = os.path.join(result_dir, 'training_history.npy')
                if os.path.exists(history_path):
                    history = np.load(history_path, allow_pickle=True).item()
                    best_val_acc = max(history['val_acc'])
                else:
                    best_val_acc = 0.0
                
                # 加载分类报告
                report_path = os.path.join(result_dir, 'classification_report.json')
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        classification_report = json.load(f)
                else:
                    classification_report = {}
                
                image_results = {
                    'method': f'Image Classification ({config["model_type"]})',
                    'model_type': config['model_type'],
                    'input_type': f'{config["image_type"]} Image',
                    'input_shape': f'{config["image_size"]}x3',
                    'accuracy': best_val_acc,
                    'parameters': 'From model',
                    'training_time': 'From logs',
                    'inference_speed': 'Medium',
                    'memory_usage': 'High',
                    'interpretability': 'High (Visual)'
                }
        
        if not image_results:
            # 默认结果（如果训练还未完成）
            image_results = {
                'method': 'Image Classification (ResNet18)',
                'model_type': 'ResNet18',
                'input_type': 'Waveform Image',
                'input_shape': '(224, 224, 3)',
                'accuracy': 0.0,  # 训练中
                'parameters': '11,179,590',
                'training_time': 'Training...',
                'inference_speed': 'Medium',
                'memory_usage': 'High',
                'interpretability': 'High (Visual)'
            }
        
        self.results['image'] = image_results
        return image_results
    
    def create_comparison_table(self):
        """创建对比表格"""
        # 加载结果
        ts_results = self.load_time_series_results()
        img_results = self.load_image_results()
        
        # 创建对比数据
        comparison_data = {
            '方法': [ts_results['method'], img_results['method']],
            '模型类型': [ts_results['model_type'], img_results['model_type']],
            '输入类型': [ts_results['input_type'], img_results['input_type']],
            '输入形状': [ts_results['input_shape'], img_results['input_shape']],
            '准确率 (%)': [ts_results['accuracy'], img_results['accuracy']],
            '参数数量': [ts_results['parameters'], img_results['parameters']],
            '推理速度': [ts_results['inference_speed'], img_results['inference_speed']],
            '内存使用': [ts_results['memory_usage'], img_results['memory_usage']],
            '可解释性': [ts_results['interpretability'], img_results['interpretability']]
        }
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_comparison_report(self, output_dir='artifacts/method_comparison'):
        """生成对比报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建对比表格
        df = self.create_comparison_table()
        
        # 保存CSV
        df.to_csv(os.path.join(output_dir, 'method_comparison.csv'), 
                 index=False, encoding='utf-8-sig')
        
        # 创建可视化
        self.create_comparison_plots(df, output_dir)
        
        # 生成文本报告
        self.generate_text_report(df, output_dir)
        
        print(f"对比报告已保存到: {output_dir}")
        return df
    
    def create_comparison_plots(self, df, output_dir):
        """创建对比图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准确率对比
        if all(isinstance(acc, (int, float)) for acc in df['准确率 (%)']):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            methods = df['方法']
            accuracies = df['准确率 (%)']
            
            bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
            ax.set_title('Method Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_xlabel('Method', fontsize=12)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
            plt.close()
        
        # 特征对比雷达图
        self.create_radar_chart(df, output_dir)
    
    def create_radar_chart(self, df, output_dir):
        """创建雷达图对比不同特征"""
        # 定义评分标准
        speed_scores = {'Fast': 5, 'Medium': 3, 'Slow': 1}
        memory_scores = {'Low': 5, 'Medium': 3, 'High': 1}  # 低内存使用得分高
        interpret_scores = {'High (Visual)': 5, 'High': 4, 'Medium': 3, 'Low': 1}
        
        # 计算评分
        methods = df['方法'].tolist()
        categories = ['Accuracy', 'Speed', 'Memory Efficiency', 'Interpretability']
        
        scores = []
        for i, method in enumerate(methods):
            acc_score = df.iloc[i]['准确率 (%)'] / 20 if isinstance(df.iloc[i]['准确率 (%)'], (int, float)) else 0  # 归一化到0-5
            speed_score = speed_scores.get(df.iloc[i]['推理速度'], 3)
            memory_score = memory_scores.get(df.iloc[i]['内存使用'], 3)
            interpret_score = interpret_scores.get(df.iloc[i]['可解释性'], 3)
            
            scores.append([acc_score, speed_score, memory_score, interpret_score])
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red']
        for i, (method, score) in enumerate(zip(methods, scores)):
            score += score[:1]  # 闭合
            ax.plot(angles, score, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, score, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_title('Method Comparison Radar Chart', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_text_report(self, df, output_dir):
        """生成文本报告"""
        report = f"""
信号波形分类方法对比报告
========================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 方法概述

### 1. 时序信号分析方法
- **输入**: 原始双通道时序信号 (500个时间步, 2个通道)
- **模型**: 1D卷积神经网络
- **优势**: 
  * 直接处理原始信号，保持时序特征
  * 计算效率高，推理速度快
  * 内存占用较少
- **劣势**:
  * 特征提取能力有限
  * 可解释性较差

### 2. 图像分类方法
- **输入**: 信号转换的波形图像 (224x224x3)
- **模型**: 预训练ResNet18
- **优势**:
  * 利用成熟的计算机视觉技术
  * 可视化效果好，便于理解和调试
  * 可以应用各种图像增强技术
  * 预训练模型提供强大的特征提取能力
- **劣势**:
  * 计算开销大，推理速度较慢
  * 内存占用高
  * 信号到图像的转换可能丢失部分信息

## 性能对比

"""
        
        # 添加表格
        report += df.to_string(index=False)
        
        report += f"""

## 结论与建议

### 适用场景分析

1. **实时应用场景**: 推荐使用时序信号方法
   - 对推理速度要求高
   - 计算资源有限
   - 内存占用敏感

2. **离线分析场景**: 可考虑图像分类方法
   - 对准确率要求高
   - 需要可视化分析
   - 计算资源充足

3. **混合方案**: 
   - 可以将两种方法结合，使用集成学习
   - 时序方法用于快速筛选，图像方法用于精确分析

### 改进建议

1. **时序方法改进**:
   - 尝试更复杂的网络结构（LSTM、Transformer等）
   - 增加特征工程
   - 使用数据增强技术

2. **图像方法改进**:
   - 尝试不同的图像表示方法（频谱图、小波变换等）
   - 优化图像尺寸和分辨率
   - 使用更轻量级的模型

3. **数据层面**:
   - 增加数据量
   - 改善数据质量
   - 平衡各类别样本数量

### 技术创新点

本项目的创新之处在于：
1. 将传统的时序信号分类问题转化为计算机视觉问题
2. 对比了两种不同范式的深度学习方法
3. 提供了完整的信号可视化和图像转换工具链

这种跨领域的方法转换为信号处理领域提供了新的思路和可能性。
"""
        
        # 保存报告
        with open(os.path.join(output_dir, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("详细对比报告已生成!")


def main():
    """主函数"""
    print("开始生成方法对比报告...")
    
    comparator = MethodComparator()
    df = comparator.generate_comparison_report()
    
    print("\n=== 方法对比表格 ===")
    print(df.to_string(index=False))
    
    print(f"\n完整报告已保存到: artifacts/method_comparison/")


if __name__ == "__main__":
    main()