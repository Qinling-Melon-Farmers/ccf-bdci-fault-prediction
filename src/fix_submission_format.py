"""
修复图像方法提交文件的标签格式
将数字标签转换为故障类型名称，符合提交要求
"""

import pandas as pd
import json
import os

def fix_image_submission():
    """修复图像方法提交文件格式"""
    
    # 数字标签到故障类型名称的映射
    # 根据label_map.json和submit_example.txt的格式要求
    label_mapping = {
        0: 'inner_broken',
        1: 'inner_wear',
        2: 'normal',  # 将normal_train160改为normal
        3: 'outer_missing', 
        4: 'roller_broken',
        5: 'roller_wear'
    }
    
    print("=== 修复图像方法提交文件格式 ===")
    
    # 读取当前的图像方法提交文件
    input_file = 'artifacts/image_submission_fixed.txt'
    output_file = 'artifacts/image_submission_final.txt'
    
    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始标签分布:")
    label_counts = df['故障类型'].value_counts().sort_index()
    for label, count in label_counts.items():
        fault_type = label_mapping.get(label, f"未知标签_{label}")
        print(f"  {label} -> {fault_type}: {count}")
    
    # 转换数字标签为故障类型名称
    df['故障类型'] = df['故障类型'].map(label_mapping)
    
    # 检查是否有未映射的标签
    unmapped = df['故障类型'].isna().sum()
    if unmapped > 0:
        print(f"警告: 发现 {unmapped} 个未映射的标签")
        print("未映射的原始标签:")
        print(df[df['故障类型'].isna()]['故障类型'].value_counts())
        return
    
    print(f"\n修复后标签分布:")
    final_counts = df['故障类型'].value_counts().sort_index()
    for fault_type, count in final_counts.items():
        print(f"  {fault_type}: {count}")
    
    # 保存修复后的文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"\n修复后的提交文件已保存: {output_file}")
    print(f"总预测数量: {len(df)}")
    
    # 验证格式
    print("\n=== 格式验证 ===")
    print("文件头部预览:")
    print(df.head(10).to_string(index=False))
    
    # 检查是否符合提交要求
    expected_columns = ['测试集名称', '故障类型']
    if list(df.columns) == expected_columns:
        print("✓ 列名格式正确")
    else:
        print(f"✗ 列名格式错误，期望: {expected_columns}, 实际: {list(df.columns)}")
    
    # 检查故障类型是否都是有效的
    valid_fault_types = {'inner_broken', 'inner_wear', 'normal', 'outer_missing', 'roller_broken', 'roller_wear'}
    actual_fault_types = set(df['故障类型'].unique())
    
    if actual_fault_types.issubset(valid_fault_types):
        print("✓ 故障类型标签正确")
    else:
        invalid_types = actual_fault_types - valid_fault_types
        print(f"✗ 发现无效的故障类型: {invalid_types}")
    
    # 检查测试集名称格式
    test_names = df['测试集名称'].tolist()
    expected_pattern = all(name.startswith('test') and name[4:].isdigit() for name in test_names[:10])
    if expected_pattern:
        print("✓ 测试集名称格式正确")
    else:
        print("✗ 测试集名称格式可能有问题")
    
    print(f"\n=== 修复完成 ===")
    return output_file

if __name__ == "__main__":
    fix_image_submission()