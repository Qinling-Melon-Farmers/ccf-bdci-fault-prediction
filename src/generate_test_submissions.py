"""
生成6份测试提交文件，每份都填满一种故障类型
用于测试测试集分布和验证提交格式
"""

import pandas as pd
import os

def generate_test_submissions():
    """生成6份测试提交文件，每份填满一种故障类型"""
    
    # 6种故障类型
    fault_types = [
        'inner_broken',
        'inner_wear', 
        'normal',
        'outer_missing',
        'roller_broken',
        'roller_wear'
    ]
    
    # 生成测试集名称列表 (test0001 到 test1140)
    test_names = [f"test{i:04d}" for i in range(1, 1141)]
    
    # 创建输出目录
    output_dir = 'artifacts/test_submissions'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 生成测试提交文件 ===")
    
    # 为每种故障类型生成一份提交文件
    for fault_type in fault_types:
        print(f"生成 {fault_type} 类型的测试文件...")
        
        # 创建DataFrame
        df = pd.DataFrame({
            '测试集名称': test_names,
            '故障类型': [fault_type] * len(test_names)
        })
        
        # 保存文件
        output_file = os.path.join(output_dir, f'test_submission_{fault_type}.txt')
        df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
        
        print(f"  ✓ 已保存: {output_file}")
        print(f"    - 测试样本数: {len(df)}")
        print(f"    - 故障类型: {fault_type}")
        
        # 显示文件头部预览
        print(f"    - 文件预览:")
        print(f"      {df.head(3).to_string(index=False)}")
        print()
    
    print("=== 生成汇总信息 ===")
    
    # 生成汇总报告
    summary_report = []
    summary_report.append("测试提交文件生成报告")
    summary_report.append("=" * 50)
    summary_report.append(f"生成时间: {pd.Timestamp.now()}")
    summary_report.append(f"输出目录: {output_dir}")
    summary_report.append(f"测试样本总数: {len(test_names)}")
    summary_report.append("")
    summary_report.append("生成的文件列表:")
    
    for i, fault_type in enumerate(fault_types, 1):
        filename = f'test_submission_{fault_type}.txt'
        summary_report.append(f"{i}. {filename}")
        summary_report.append(f"   - 故障类型: {fault_type}")
        summary_report.append(f"   - 样本数量: {len(test_names)}")
        summary_report.append("")
    
    summary_report.append("文件格式说明:")
    summary_report.append("- 列分隔符: Tab (\\t)")
    summary_report.append("- 编码格式: UTF-8")
    summary_report.append("- 列名: ['测试集名称', '故障类型']")
    summary_report.append("- 测试集名称格式: test0001 ~ test1140")
    summary_report.append("")
    summary_report.append("用途说明:")
    summary_report.append("- 测试测试集分布")
    summary_report.append("- 验证提交文件格式")
    summary_report.append("- 分析各故障类型的预测效果")
    
    # 保存汇总报告
    summary_file = os.path.join(output_dir, 'generation_report.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_report))
    
    print(f"汇总报告已保存: {summary_file}")
    
    # 验证生成的文件
    print("\n=== 验证生成的文件 ===")
    for fault_type in fault_types:
        filename = f'test_submission_{fault_type}.txt'
        filepath = os.path.join(output_dir, filename)
        
        # 检查文件是否存在
        if os.path.exists(filepath):
            # 读取并验证文件
            df_check = pd.read_csv(filepath, sep='\t')
            
            # 验证格式
            expected_columns = ['测试集名称', '故障类型']
            if list(df_check.columns) == expected_columns:
                print(f"✓ {filename}: 格式正确")
            else:
                print(f"✗ {filename}: 格式错误")
            
            # 验证数据量
            if len(df_check) == 1140:
                print(f"✓ {filename}: 数据量正确 (1140行)")
            else:
                print(f"✗ {filename}: 数据量错误 ({len(df_check)}行)")
            
            # 验证故障类型一致性
            unique_types = df_check['故障类型'].unique()
            if len(unique_types) == 1 and unique_types[0] == fault_type:
                print(f"✓ {filename}: 故障类型一致 ({fault_type})")
            else:
                print(f"✗ {filename}: 故障类型不一致 ({unique_types})")
        else:
            print(f"✗ {filename}: 文件不存在")
        
        print()
    
    print("=== 测试提交文件生成完成 ===")
    print(f"所有文件已保存到: {output_dir}")
    print("可以使用这些文件测试测试集分布和验证提交格式")

if __name__ == "__main__":
    generate_test_submissions()