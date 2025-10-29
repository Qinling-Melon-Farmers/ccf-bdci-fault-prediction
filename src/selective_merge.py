"""
择优合并三种方法的预测结果
基于分析结果，在roller_wear类别优先使用树模型，其他类别选择最佳方法
"""

import pandas as pd
import numpy as np
from collections import Counter
import os


class SelectiveMerger:
    """择优合并器"""
    
    def __init__(self):
        self.fault_types = ['normal', 'inner_wear', 'outer_missing', 'roller_wear', 'inner_broken', 'roller_broken']
    
    def load_all_predictions(self):
        """加载所有方法的预测结果"""
        
        print("=== 加载预测结果 ===")
        
        # 加载树模型结果
        tree_df = pd.read_csv('enhanced_submit_result.txt', sep='\t')
        print(f"树模型预测数量: {len(tree_df)}")
        
        # 加载时序方法结果
        ts_df = pd.read_csv('artifacts/simple_submission.txt', sep='\t')
        print(f"时序方法预测数量: {len(ts_df)}")
        
        # 加载图像方法结果
        img_df = pd.read_csv('artifacts/image_submission_final.txt', sep='\t')
        print(f"图像方法预测数量: {len(img_df)}")
        
        # 验证数据一致性
        assert len(tree_df) == len(ts_df) == len(img_df), "三种方法的预测数量不一致"
        
        # 合并所有预测结果
        merged = pd.merge(tree_df[['测试集名称', '故障类型']], 
                         ts_df[['测试集名称', '故障类型']], 
                         on='测试集名称', suffixes=('_tree', '_ts'))
        merged = pd.merge(merged, 
                         img_df[['测试集名称', '故障类型']], 
                         on='测试集名称')
        merged.rename(columns={'故障类型': '故障类型_img'}, inplace=True)
        
        print(f"合并后数据量: {len(merged)}")
        return merged
    
    def analyze_method_strengths(self, merged):
        """分析各方法在不同类别上的优势"""
        
        print("\n=== 分析各方法优势 ===")
        
        # 基于分析报告的结果，定义各方法的优势类别
        method_strengths = {
            'tree': {
                'strong': ['roller_wear', 'roller_broken'],  # 树模型在roller相关故障上表现好
                'medium': ['outer_missing', 'inner_broken'],
                'weak': ['normal', 'inner_wear']
            },
            'ts': {
                'strong': ['normal', 'outer_missing'],  # 时序方法在normal和outer_missing上表现好
                'medium': ['roller_wear', 'inner_wear'],
                'weak': ['roller_broken', 'inner_broken']
            },
            'img': {
                'strong': ['inner_wear'],  # 图像方法在inner_wear上预测较多
                'medium': ['normal', 'roller_wear'],
                'weak': ['outer_missing', 'roller_broken', 'inner_broken']
            }
        }
        
        return method_strengths
    
    def implement_selective_merge_strategy(self, merged, method_strengths):
        """实现择优合并策略"""
        
        print("\n=== 实施择优合并策略 ===")
        
        final_predictions = []
        merge_stats = {
            'tree_selected': 0,
            'ts_selected': 0,
            'img_selected': 0,
            'by_category': {}
        }
        
        for _, row in merged.iterrows():
            test_name = row['测试集名称']
            tree_pred = row['故障类型_tree']
            ts_pred = row['故障类型_ts']
            img_pred = row['故障类型_img']
            
            # 策略1: roller_wear优先使用树模型（基于用户反馈）
            if tree_pred == 'roller_wear':
                final_pred = tree_pred
                selected_method = 'tree'
                reason = 'tree_roller_wear_priority'
            
            # 策略2: roller_broken优先使用树模型（树模型在此类别预测最多）
            elif tree_pred == 'roller_broken':
                final_pred = tree_pred
                selected_method = 'tree'
                reason = 'tree_roller_broken_strength'
            
            # 策略3: outer_missing类别，树模型和时序方法都表现好，选择一致的
            elif tree_pred == 'outer_missing' or ts_pred == 'outer_missing':
                if tree_pred == ts_pred == 'outer_missing':
                    final_pred = 'outer_missing'
                    selected_method = 'consensus'
                    reason = 'tree_ts_consensus_outer_missing'
                elif tree_pred == 'outer_missing':
                    final_pred = tree_pred
                    selected_method = 'tree'
                    reason = 'tree_outer_missing_strength'
                else:
                    final_pred = ts_pred
                    selected_method = 'ts'
                    reason = 'ts_outer_missing_strength'
            
            # 策略4: normal类别，时序方法表现最好
            elif ts_pred == 'normal':
                final_pred = ts_pred
                selected_method = 'ts'
                reason = 'ts_normal_strength'
            
            # 策略5: inner_wear类别，考虑图像方法的预测
            elif img_pred == 'inner_wear':
                # 如果树模型或时序方法也预测为inner_wear，选择一致的
                if tree_pred == 'inner_wear' or ts_pred == 'inner_wear':
                    final_pred = 'inner_wear'
                    selected_method = 'consensus'
                    reason = 'inner_wear_consensus'
                else:
                    # 图像方法单独预测inner_wear，但考虑其他方法的意见
                    final_pred = img_pred
                    selected_method = 'img'
                    reason = 'img_inner_wear_strength'
            
            # 策略6: inner_broken类别，优先考虑时序方法
            elif ts_pred == 'inner_broken':
                final_pred = ts_pred
                selected_method = 'ts'
                reason = 'ts_inner_broken_strength'
            elif tree_pred == 'inner_broken':
                final_pred = tree_pred
                selected_method = 'tree'
                reason = 'tree_inner_broken_backup'
            
            # 策略7: 默认情况，选择时序方法（整体表现最好）
            else:
                final_pred = ts_pred
                selected_method = 'ts'
                reason = 'ts_default'
            
            # 记录统计信息
            if selected_method == 'tree':
                merge_stats['tree_selected'] += 1
            elif selected_method == 'ts':
                merge_stats['ts_selected'] += 1
            elif selected_method == 'img':
                merge_stats['img_selected'] += 1
            
            if final_pred not in merge_stats['by_category']:
                merge_stats['by_category'][final_pred] = {'tree': 0, 'ts': 0, 'img': 0, 'consensus': 0}
            
            if selected_method == 'consensus':
                merge_stats['by_category'][final_pred]['consensus'] += 1
            else:
                merge_stats['by_category'][final_pred][selected_method] += 1
            
            final_predictions.append({
                '测试集名称': test_name,
                '故障类型': final_pred,
                'selected_method': selected_method,
                'reason': reason,
                'tree_pred': tree_pred,
                'ts_pred': ts_pred,
                'img_pred': img_pred
            })
        
        return final_predictions, merge_stats
    
    def analyze_merge_results(self, final_predictions, merge_stats):
        """分析合并结果"""
        
        print("\n=== 合并结果分析 ===")
        
        # 统计最终预测分布
        final_df = pd.DataFrame(final_predictions)
        final_dist = final_df['故障类型'].value_counts().sort_index()
        
        print("最终预测分布:")
        for fault_type, count in final_dist.items():
            percentage = count / len(final_df) * 100
            print(f"  {fault_type}: {count} ({percentage:.1f}%)")
        
        # 统计方法选择情况
        print(f"\n方法选择统计:")
        print(f"  树模型被选择: {merge_stats['tree_selected']} ({merge_stats['tree_selected']/len(final_df)*100:.1f}%)")
        print(f"  时序方法被选择: {merge_stats['ts_selected']} ({merge_stats['ts_selected']/len(final_df)*100:.1f}%)")
        print(f"  图像方法被选择: {merge_stats['img_selected']} ({merge_stats['img_selected']/len(final_df)*100:.1f}%)")
        
        # 按类别统计方法选择
        print(f"\n各类别方法选择详情:")
        for fault_type, methods in merge_stats['by_category'].items():
            total = sum(methods.values())
            print(f"  {fault_type} (总计{total}):")
            for method, count in methods.items():
                if count > 0:
                    print(f"    {method}: {count} ({count/total*100:.1f}%)")
        
        return final_df
    
    def save_merged_results(self, final_df, output_path='artifacts/selective_merged_submission.txt'):
        """保存合并结果"""
        
        print(f"\n=== 保存合并结果 ===")
        
        # 创建提交格式的文件
        submission_df = final_df[['测试集名称', '故障类型']].copy()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存提交文件
        submission_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"合并后的提交文件已保存: {output_path}")
        
        # 保存详细分析文件
        detail_path = output_path.replace('.txt', '_detailed.csv')
        final_df.to_csv(detail_path, index=False, encoding='utf-8')
        print(f"详细分析文件已保存: {detail_path}")
        
        # 验证文件格式
        self.validate_submission_format(output_path)
        
        return output_path
    
    def validate_submission_format(self, file_path):
        """验证提交文件格式"""
        
        print(f"\n=== 验证提交文件格式 ===")
        
        df = pd.read_csv(file_path, sep='\t')
        
        # 检查列名
        expected_columns = ['测试集名称', '故障类型']
        if list(df.columns) != expected_columns:
            print(f"❌ 列名不正确: {list(df.columns)}, 期望: {expected_columns}")
            return False
        
        # 检查数据量
        if len(df) != 1140:
            print(f"❌ 数据量不正确: {len(df)}, 期望: 1140")
            return False
        
        # 检查故障类型
        valid_fault_types = {'normal', 'inner_wear', 'outer_missing', 'roller_wear', 'inner_broken', 'roller_broken'}
        invalid_types = set(df['故障类型'].unique()) - valid_fault_types
        if invalid_types:
            print(f"❌ 包含无效故障类型: {invalid_types}")
            return False
        
        # 检查测试集名称格式
        test_names = df['测试集名称'].tolist()
        expected_names = [f'test{i:04d}' for i in range(1, 1141)]
        if test_names != expected_names:
            print("❌ 测试集名称格式不正确")
            return False
        
        print("✅ 提交文件格式验证通过")
        return True
    
    def generate_merge_report(self, final_df, merge_stats, output_dir='artifacts'):
        """生成合并报告"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计最终分布
        final_dist = final_df['故障类型'].value_counts().sort_index()
        
        report = f"""
择优合并结果报告
===============

## 合并策略

基于三种方法的分析结果，采用以下择优合并策略:

1. **roller_wear类别**: 优先使用树模型（用户反馈极佳准确性）
2. **roller_broken类别**: 优先使用树模型（树模型预测数量最多）
3. **outer_missing类别**: 树模型和时序方法一致时选择一致结果，否则选择各自优势
4. **normal类别**: 优先使用时序方法（时序方法表现最好）
5. **inner_wear类别**: 考虑图像方法的预测，结合其他方法意见
6. **inner_broken类别**: 优先使用时序方法
7. **默认情况**: 选择时序方法（整体表现最好）

## 合并结果统计

### 最终预测分布
"""
        
        for fault_type, count in final_dist.items():
            percentage = count / len(final_df) * 100
            report += f"- {fault_type}: {count} ({percentage:.1f}%)\n"
        
        report += f"""

### 方法选择统计
- 树模型被选择: {merge_stats['tree_selected']} ({merge_stats['tree_selected']/len(final_df)*100:.1f}%)
- 时序方法被选择: {merge_stats['ts_selected']} ({merge_stats['ts_selected']/len(final_df)*100:.1f}%)
- 图像方法被选择: {merge_stats['img_selected']} ({merge_stats['img_selected']/len(final_df)*100:.1f}%)

### 各类别方法选择详情
"""
        
        for fault_type, methods in merge_stats['by_category'].items():
            total = sum(methods.values())
            report += f"\n**{fault_type}** (总计{total}):\n"
            for method, count in methods.items():
                if count > 0:
                    report += f"  - {method}: {count} ({count/total*100:.1f}%)\n"
        
        report += f"""

## 关键优化点

1. **roller_wear优势**: 树模型在roller_wear类别的预测被优先采用，共{merge_stats['by_category'].get('roller_wear', {}).get('tree', 0)}个样本
2. **时序方法主导**: 时序方法作为默认选择，在多个类别中表现稳定
3. **图像方法补充**: 图像方法在inner_wear类别提供有价值的预测

## 预期效果

通过择优合并，预期能够:
- 充分利用树模型在roller_wear上的优势
- 保持时序方法的整体稳定性
- 在特定类别上获得更好的预测效果

## 文件输出

- 提交文件: artifacts/selective_merged_submission.txt
- 详细分析: artifacts/selective_merged_submission_detailed.csv
- 本报告: artifacts/selective_merge_report.txt
"""
        
        # 保存报告
        report_path = os.path.join(output_dir, 'selective_merge_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"合并报告已保存: {report_path}")


def main():
    """主函数"""
    print("=== 开始择优合并三种方法的预测结果 ===")
    
    merger = SelectiveMerger()
    
    # 加载所有预测结果
    merged = merger.load_all_predictions()
    
    # 分析各方法优势
    method_strengths = merger.analyze_method_strengths(merged)
    
    # 实施择优合并策略
    final_predictions, merge_stats = merger.implement_selective_merge_strategy(merged, method_strengths)
    
    # 分析合并结果
    final_df = merger.analyze_merge_results(final_predictions, merge_stats)
    
    # 保存合并结果
    output_path = merger.save_merged_results(final_df)
    
    # 生成合并报告
    merger.generate_merge_report(final_df, merge_stats)
    
    print(f"\n=== 择优合并完成 ===")
    print(f"最终提交文件: {output_path}")
    print("详细结果已保存到 artifacts/ 目录")


if __name__ == "__main__":
    main()