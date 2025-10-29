"""
对比时序方法和图像方法的测试集预测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os


class PredictionComparator:
    """预测结果对比器"""
    
    def __init__(self):
        # 类别映射
        self.class_names = {
            0: 'normal',
            1: 'inner_wear', 
            2: 'outer_missing',
            3: 'roller_wear',
            4: 'inner_broken',
            5: 'roller_broken'
        }
        
        self.name_to_id = {v: k for k, v in self.class_names.items()}
    
    def load_predictions(self):
        """加载两种方法的预测结果"""
        
        # 加载时序方法结果
        ts_df = pd.read_csv('artifacts/simple_submission.txt', sep='\t')
        ts_df['method'] = 'Time Series'
        ts_df['class_id'] = ts_df['故障类型'].map(self.name_to_id)
        
        # 加载图像方法结果
        img_df = pd.read_csv('artifacts/image_submission_fixed.txt', sep='\t')
        img_df['method'] = 'Image Classification'
        img_df['class_id'] = img_df['故障类型']
        img_df['故障类型'] = img_df['故障类型'].map(self.class_names)
        
        print(f"时序方法预测数量: {len(ts_df)}")
        print(f"图像方法预测数量: {len(img_df)}")
        
        return ts_df, img_df
    
    def compare_distributions(self, ts_df, img_df):
        """对比预测分布"""
        print("\n=== 预测分布对比 ===")
        
        # 时序方法分布
        ts_dist = ts_df['故障类型'].value_counts().sort_index()
        print("\n时序方法预测分布:")
        for class_name, count in ts_dist.items():
            print(f"  {class_name}: {count} ({count/len(ts_df)*100:.1f}%)")
        
        # 图像方法分布
        img_dist = img_df['故障类型'].value_counts().sort_index()
        print("\n图像方法预测分布:")
        for class_name, count in img_dist.items():
            print(f"  {class_name}: {count} ({count/len(img_df)*100:.1f}%)")
        
        return ts_dist, img_dist
    
    def analyze_agreement(self, ts_df, img_df):
        """分析两种方法的一致性"""
        print("\n=== 预测一致性分析 ===")
        
        # 合并数据
        merged = pd.merge(ts_df[['测试集名称', '故障类型', 'class_id']], 
                         img_df[['测试集名称', '故障类型', 'class_id']], 
                         on='测试集名称', suffixes=('_ts', '_img'))
        
        # 计算一致性
        agreement = (merged['class_id_ts'] == merged['class_id_img']).sum()
        total = len(merged)
        agreement_rate = agreement / total * 100
        
        print(f"预测一致的样本数: {agreement}/{total}")
        print(f"一致性比例: {agreement_rate:.2f}%")
        
        # 分析不一致的情况
        disagreement = merged[merged['class_id_ts'] != merged['class_id_img']]
        print(f"预测不一致的样本数: {len(disagreement)}")
        
        if len(disagreement) > 0:
            print("\n不一致样本的类别分布:")
            disagree_pairs = disagreement.groupby(['故障类型_ts', '故障类型_img']).size().sort_values(ascending=False)
            for (ts_class, img_class), count in disagree_pairs.head(10).items():
                print(f"  时序:{ts_class} vs 图像:{img_class} -> {count}次")
        
        return merged, agreement_rate
    
    def create_comparison_plots(self, ts_dist, img_dist, output_dir='artifacts/prediction_comparison'):
        """创建对比图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 预测分布对比柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 时序方法分布
        ts_dist.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
        ax1.set_title('Time Series Method Predictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_xlabel('Fault Type', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 图像方法分布
        img_dist.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.7)
        ax2.set_title('Image Classification Method Predictions', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_xlabel('Fault Type', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 对比饼图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 时序方法饼图
        ts_dist.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Time Series Method Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('')
        
        # 图像方法饼图
        img_dist.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Image Classification Method Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_pie_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存到: {output_dir}")
    
    def create_confusion_matrix(self, merged, output_dir='artifacts/prediction_comparison'):
        """创建混淆矩阵显示两种方法的对比"""
        
        # 创建混淆矩阵数据
        confusion_data = pd.crosstab(merged['故障类型_ts'], merged['故障类型_img'], margins=True)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_data.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues',
                   xticklabels=confusion_data.columns[:-1], 
                   yticklabels=confusion_data.index[:-1])
        plt.title('Prediction Agreement Matrix\n(Time Series vs Image Classification)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Image Classification Predictions', fontsize=12)
        plt.ylabel('Time Series Predictions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, ts_df, img_df, merged, agreement_rate, 
                                 output_dir='artifacts/prediction_comparison'):
        """生成详细的对比报告"""
        
        report = f"""
测试集预测结果对比报告
====================

## 基本信息
- 测试样本总数: {len(merged)}
- 时序方法预测数: {len(ts_df)}
- 图像方法预测数: {len(img_df)}
- 预测一致性: {agreement_rate:.2f}%

## 预测分布对比

### 时序方法 (1D CNN)
"""
        
        ts_dist = ts_df['故障类型'].value_counts().sort_index()
        for class_name, count in ts_dist.items():
            percentage = count/len(ts_df)*100
            report += f"- {class_name}: {count} ({percentage:.1f}%)\n"
        
        report += "\n### 图像方法 (ResNet18)\n"
        img_dist = img_df['故障类型'].value_counts().sort_index()
        for class_name, count in img_dist.items():
            percentage = count/len(img_df)*100
            report += f"- {class_name}: {count} ({percentage:.1f}%)\n"
        
        # 分析差异
        report += f"""

## 方法差异分析

### 预测一致性
- 两种方法预测完全一致的样本: {(merged['class_id_ts'] == merged['class_id_img']).sum()}
- 预测不一致的样本: {(merged['class_id_ts'] != merged['class_id_img']).sum()}
- 一致性比例: {agreement_rate:.2f}%

### 主要差异
"""
        
        # 找出最大的预测差异
        disagreement = merged[merged['class_id_ts'] != merged['class_id_img']]
        if len(disagreement) > 0:
            disagree_pairs = disagreement.groupby(['故障类型_ts', '故障类型_img']).size().sort_values(ascending=False)
            for (ts_class, img_class), count in disagree_pairs.head(5).items():
                report += f"- 时序方法预测'{ts_class}' vs 图像方法预测'{img_class}': {count}次\n"
        
        report += f"""

## 方法特点总结

### 时序方法 (1D CNN)
- 验证准确率: 36.34%
- 预测特点: 直接处理原始信号，保持时序特征
- 计算效率: 高，推理速度快
- 内存占用: 低

### 图像方法 (ResNet18)  
- 验证准确率: 34.01%
- 预测特点: 将信号转换为图像，利用视觉特征
- 计算效率: 中等，需要图像转换和CNN推理
- 内存占用: 高

## 结论

1. **性能对比**: 两种方法在验证集上的准确率相近，时序方法略高
2. **预测一致性**: {agreement_rate:.1f}%的预测一致性表明两种方法有相似的判断倾向
3. **互补性**: 预测不一致的样本可能是困难样本，可以考虑集成学习
4. **应用场景**: 
   - 实时应用: 推荐时序方法
   - 离线分析: 可考虑图像方法或集成方法

## 改进建议

1. **集成学习**: 结合两种方法的预测结果
2. **困难样本分析**: 重点分析预测不一致的样本
3. **特征融合**: 结合时序特征和图像特征
4. **模型优化**: 进一步调优两种方法的超参数
"""
        
        # 保存报告
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'prediction_comparison_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"详细对比报告已保存: {output_dir}/prediction_comparison_report.txt")


def main():
    """主函数"""
    print("=== 测试集预测结果对比分析 ===")
    
    comparator = PredictionComparator()
    
    # 加载预测结果
    ts_df, img_df = comparator.load_predictions()
    
    # 对比预测分布
    ts_dist, img_dist = comparator.compare_distributions(ts_df, img_df)
    
    # 分析一致性
    merged, agreement_rate = comparator.analyze_agreement(ts_df, img_df)
    
    # 创建对比图表
    comparator.create_comparison_plots(ts_dist, img_dist)
    comparator.create_confusion_matrix(merged)
    
    # 生成详细报告
    comparator.generate_comparison_report(ts_df, img_df, merged, agreement_rate)
    
    print(f"\n=== 对比分析完成 ===")
    print(f"预测一致性: {agreement_rate:.2f}%")
    print("详细结果已保存到: artifacts/prediction_comparison/")


if __name__ == "__main__":
    main()