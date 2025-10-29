"""
修复图像方法提交文件格式
处理重复的文件ID，生成正确的提交格式
"""

import numpy as np
import pandas as pd
from collections import Counter


def fix_image_submission():
    """修复图像方法的提交文件"""
    
    # 加载测试数据
    test_data = np.load('artifacts/datasets/test.npz')
    file_ids = test_data['file_ids']
    
    # 加载预测结果
    df = pd.read_csv('artifacts/image_submission.txt', sep='\t')
    
    print(f"原始预测结果数量: {len(df)}")
    print(f"文件ID数量: {len(file_ids)}")
    print(f"唯一文件ID数量: {len(set(file_ids))}")
    
    # 分析文件ID重复情况
    file_id_counts = Counter(file_ids)
    print(f"每个文件ID出现次数: {list(file_id_counts.values())[:10]}")
    
    # 获取唯一的文件ID
    unique_file_ids = []
    seen = set()
    for fid in file_ids:
        if fid not in seen:
            unique_file_ids.append(fid)
            seen.add(fid)
    
    print(f"唯一文件ID数量: {len(unique_file_ids)}")
    
    # 重新组织预测结果
    # 每个文件有两个样本（两个通道），我们需要合并它们的预测
    fixed_predictions = []
    
    i = 0
    for unique_id in unique_file_ids:
        # 找到该文件ID对应的所有预测
        file_predictions = []
        while i < len(df) and df.iloc[i]['测试集名称'] == unique_id:
            file_predictions.append(df.iloc[i]['故障类型'])
            i += 1
        
        # 使用多数投票或第一个预测作为最终预测
        if len(file_predictions) > 1:
            # 多数投票
            pred_counts = Counter(file_predictions)
            final_pred = pred_counts.most_common(1)[0][0]
        else:
            final_pred = file_predictions[0]
        
        fixed_predictions.append({
            '测试集名称': unique_id,
            '故障类型': final_pred
        })
    
    # 创建修复后的DataFrame
    fixed_df = pd.DataFrame(fixed_predictions)
    
    # 保存修复后的提交文件
    fixed_df.to_csv('artifacts/image_submission_fixed.txt', sep='\t', index=False, encoding='utf-8')
    
    print(f"修复后的预测结果数量: {len(fixed_df)}")
    print("预测结果统计:")
    print(fixed_df['故障类型'].value_counts().sort_index())
    
    print("修复后的提交文件已保存: artifacts/image_submission_fixed.txt")
    
    return fixed_df


def main():
    """主函数"""
    print("=== 修复图像方法提交文件格式 ===")
    
    fixed_df = fix_image_submission()
    
    # 显示前几行
    print("\n前10行预测结果:")
    print(fixed_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()