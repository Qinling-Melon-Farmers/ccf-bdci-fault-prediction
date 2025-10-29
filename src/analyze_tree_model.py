"""
分析树模型预测结果的分布和特点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os


def analyze_tree_model_predictions():
    """分析树模型预测结果"""
    
    print("=== 树模型预测结果分析 ===")
    
    # 加载树模型预测结果
    tree_df = pd.read_csv('enhanced_submit_result.txt', sep='\t')
    print(f"树模型预测样本数: {len(tree_df)}")
    
    # 分析预测分布
    print("\n=== 树模型预测分布 ===")
    tree_dist = tree_df['故障类型'].value_counts().sort_index()
    total_samples = len(tree_df)
    
    for class_name, count in tree_dist.items():
        percentage = count / total_samples * 100
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    # 加载其他方法的预测结果进行对比
    print("\n=== 加载其他方法预测结果 ===")
    
    # 时序方法
    ts_df = pd.read_csv('artifacts/simple_submission.txt', sep='\t')
    print(f"时序方法预测样本数: {len(ts_df)}")
    
    # 图像方法
    img_df = pd.read_csv('artifacts/image_submission_final.txt', sep='\t')
    print(f"图像方法预测样本数: {len(img_df)}")
    
    return tree_df, ts_df, img_df


def compare_class_predictions(tree_df, ts_df, img_df):
    """对比各方法在不同类别上的预测分布"""
    
    print("\n=== 各方法预测分布对比 ===")
    
    # 获取各方法的分布
    tree_dist = tree_df['故障类型'].value_counts().sort_index()
    ts_dist = ts_df['故障类型'].value_counts().sort_index()
    img_dist = img_df['故障类型'].value_counts().sort_index()
    
    # 创建对比表格
    comparison_df = pd.DataFrame({
        '树模型': tree_dist,
        '时序方法': ts_dist,
        '图像方法': img_dist
    }).fillna(0).astype(int)
    
    # 计算百分比
    comparison_pct = pd.DataFrame({
        '树模型(%)': (tree_dist / len(tree_df) * 100).round(1),
        '时序方法(%)': (ts_dist / len(ts_df) * 100).round(1),
        '图像方法(%)': (img_dist / len(img_df) * 100).round(1)
    }).fillna(0)
    
    print("\n预测数量对比:")
    print(comparison_df)
    
    print("\n预测百分比对比:")
    print(comparison_pct)
    
    # 特别关注roller_wear类别
    print(f"\n=== roller_wear类别详细分析 ===")
    roller_wear_counts = {
        '树模型': tree_dist.get('roller_wear', 0),
        '时序方法': ts_dist.get('roller_wear', 0),
        '图像方法': img_dist.get('roller_wear', 0)
    }
    
    for method, count in roller_wear_counts.items():
        total = len(tree_df) if method == '树模型' else (len(ts_df) if method == '时序方法' else len(img_df))
        percentage = count / total * 100
        print(f"{method} roller_wear预测: {count} ({percentage:.1f}%)")
    
    return comparison_df, comparison_pct


def analyze_prediction_agreement(tree_df, ts_df, img_df):
    """分析各方法之间的预测一致性"""
    
    print("\n=== 预测一致性分析 ===")
    
    # 合并所有预测结果
    merged = pd.merge(tree_df[['测试集名称', '故障类型']], 
                     ts_df[['测试集名称', '故障类型']], 
                     on='测试集名称', suffixes=('_tree', '_ts'))
    merged = pd.merge(merged, 
                     img_df[['测试集名称', '故障类型']], 
                     on='测试集名称')
    merged.rename(columns={'故障类型': '故障类型_img'}, inplace=True)
    
    total_samples = len(merged)
    
    # 计算两两一致性
    tree_ts_agree = (merged['故障类型_tree'] == merged['故障类型_ts']).sum()
    tree_img_agree = (merged['故障类型_tree'] == merged['故障类型_img']).sum()
    ts_img_agree = (merged['故障类型_ts'] == merged['故障类型_img']).sum()
    
    print(f"树模型 vs 时序方法一致性: {tree_ts_agree}/{total_samples} ({tree_ts_agree/total_samples*100:.1f}%)")
    print(f"树模型 vs 图像方法一致性: {tree_img_agree}/{total_samples} ({tree_img_agree/total_samples*100:.1f}%)")
    print(f"时序方法 vs 图像方法一致性: {ts_img_agree}/{total_samples} ({ts_img_agree/total_samples*100:.1f}%)")
    
    # 三方法完全一致
    all_agree = ((merged['故障类型_tree'] == merged['故障类型_ts']) & 
                (merged['故障类型_ts'] == merged['故障类型_img'])).sum()
    print(f"三种方法完全一致: {all_agree}/{total_samples} ({all_agree/total_samples*100:.1f}%)")
    
    # 分析roller_wear类别的一致性
    roller_wear_samples = merged[
        (merged['故障类型_tree'] == 'roller_wear') |
        (merged['故障类型_ts'] == 'roller_wear') |
        (merged['故障类型_img'] == 'roller_wear')
    ]
    
    print(f"\n=== roller_wear类别一致性分析 ===")
    print(f"至少一个方法预测为roller_wear的样本数: {len(roller_wear_samples)}")
    
    if len(roller_wear_samples) > 0:
        # 在roller_wear相关样本中的一致性
        rw_tree_ts = (roller_wear_samples['故障类型_tree'] == roller_wear_samples['故障类型_ts']).sum()
        rw_tree_img = (roller_wear_samples['故障类型_tree'] == roller_wear_samples['故障类型_img']).sum()
        rw_ts_img = (roller_wear_samples['故障类型_ts'] == roller_wear_samples['故障类型_img']).sum()
        
        print(f"在roller_wear相关样本中:")
        print(f"  树模型 vs 时序方法: {rw_tree_ts}/{len(roller_wear_samples)} ({rw_tree_ts/len(roller_wear_samples)*100:.1f}%)")
        print(f"  树模型 vs 图像方法: {rw_tree_img}/{len(roller_wear_samples)} ({rw_tree_img/len(roller_wear_samples)*100:.1f}%)")
        print(f"  时序方法 vs 图像方法: {rw_ts_img}/{len(roller_wear_samples)} ({rw_ts_img/len(roller_wear_samples)*100:.1f}%)")
    
    return merged


def create_comparison_visualization(comparison_df, output_dir='artifacts/tree_model_analysis'):
    """创建对比可视化图表"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建对比柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(comparison_df.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, comparison_df['树模型'], width, label='树模型', alpha=0.8, color='green')
    bars2 = ax.bar(x, comparison_df['时序方法'], width, label='时序方法', alpha=0.8, color='skyblue')
    bars3 = ax.bar(x + width, comparison_df['图像方法'], width, label='图像方法', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('故障类型', fontsize=12)
    ax.set_ylabel('预测数量', fontsize=12)
    ax.set_title('三种方法预测分布对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend()
    
    # 在柱状图上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'methods_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {output_dir}/methods_comparison.png")


def generate_analysis_report(tree_df, comparison_df, merged, output_dir='artifacts/tree_model_analysis'):
    """生成分析报告"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算一致性统计
    total_samples = len(merged)
    tree_ts_agree = (merged['故障类型_tree'] == merged['故障类型_ts']).sum()
    tree_img_agree = (merged['故障类型_tree'] == merged['故障类型_img']).sum()
    ts_img_agree = (merged['故障类型_ts'] == merged['故障类型_img']).sum()
    
    report = f"""
树模型预测结果分析报告
===================

## 基本信息
- 测试样本总数: {len(tree_df)}
- 分析的方法: 树模型、时序方法(1D CNN)、图像方法(ResNet18)

## 预测分布对比

### 各方法预测数量
"""
    
    for fault_type in comparison_df.index:
        tree_count = comparison_df.loc[fault_type, '树模型']
        ts_count = comparison_df.loc[fault_type, '时序方法']
        img_count = comparison_df.loc[fault_type, '图像方法']
        
        report += f"""
{fault_type}:
  - 树模型: {tree_count} ({tree_count/len(tree_df)*100:.1f}%)
  - 时序方法: {ts_count} ({ts_count/len(tree_df)*100:.1f}%)
  - 图像方法: {img_count} ({img_count/len(tree_df)*100:.1f}%)"""
    
    # 特别分析roller_wear
    tree_roller_wear = comparison_df.loc['roller_wear', '树模型']
    ts_roller_wear = comparison_df.loc['roller_wear', '时序方法']
    img_roller_wear = comparison_df.loc['roller_wear', '图像方法']
    
    report += f"""

## roller_wear类别特别分析

根据用户反馈，树模型在roller_wear预测上有极佳的准确性。

### roller_wear预测统计:
- 树模型: {tree_roller_wear} ({tree_roller_wear/len(tree_df)*100:.1f}%)
- 时序方法: {ts_roller_wear} ({ts_roller_wear/len(tree_df)*100:.1f}%)
- 图像方法: {img_roller_wear} ({img_roller_wear/len(tree_df)*100:.1f}%)

### 观察结果:
- 树模型预测roller_wear的比例相对{"较高" if tree_roller_wear > max(ts_roller_wear, img_roller_wear) else "适中"}
- 这表明树模型可能在识别roller_wear故障方面有独特优势

## 方法间一致性分析

### 整体一致性:
- 树模型 vs 时序方法: {tree_ts_agree}/{total_samples} ({tree_ts_agree/total_samples*100:.1f}%)
- 树模型 vs 图像方法: {tree_img_agree}/{total_samples} ({tree_img_agree/total_samples*100:.1f}%)
- 时序方法 vs 图像方法: {ts_img_agree}/{total_samples} ({ts_img_agree/total_samples*100:.1f}%)

## 择优合并建议

基于分析结果，建议采用以下择优合并策略:

1. **roller_wear类别**: 优先使用树模型预测（基于用户反馈的极佳准确性）
2. **其他类别**: 根据各方法的表现选择最优预测

### 具体策略:
- 当树模型预测为roller_wear时，直接采用树模型结果
- 当树模型预测为其他类别时，可以考虑与时序方法对比选择
- 图像方法可作为辅助参考

## 实施建议

1. 实现择优合并算法
2. 重点验证roller_wear类别的合并效果
3. 对比合并前后的整体预测分布
4. 确保最终提交文件格式正确
"""
    
    # 保存报告
    with open(os.path.join(output_dir, 'tree_model_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已保存: {output_dir}/tree_model_analysis_report.txt")


def main():
    """主函数"""
    print("开始分析树模型预测结果...")
    
    # 分析树模型预测结果
    tree_df, ts_df, img_df = analyze_tree_model_predictions()
    
    # 对比各方法预测分布
    comparison_df, comparison_pct = compare_class_predictions(tree_df, ts_df, img_df)
    
    # 分析预测一致性
    merged = analyze_prediction_agreement(tree_df, ts_df, img_df)
    
    # 创建可视化图表
    create_comparison_visualization(comparison_df)
    
    # 生成分析报告
    generate_analysis_report(tree_df, comparison_df, merged)
    
    print("\n=== 树模型分析完成 ===")
    print("详细结果已保存到: artifacts/tree_model_analysis/")


if __name__ == "__main__":
    main()