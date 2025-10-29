"""
测试集推理脚本
加载训练好的模型对测试集进行预测，并生成提交文件
"""
import os
import json
import numpy as np
import torch
from collections import defaultdict

from config import DATASETS_DIR, MODELS_DIR, SUBMIT_PATH
from models.cnn1d import CNN1D


def majority_vote(preds):
    """
    多数投票函数，用于聚合同一文件的多个预测结果
    
    Args:
        preds: 预测结果数组
        
    Returns:
        int: 出现次数最多的预测类别
    """
    values, counts = np.unique(preds, return_counts=True)
    return values[np.argmax(counts)]


def main():
    """主函数：执行测试集推理和提交文件生成"""
    # ==================== 数据准备 ====================
    test_npz = os.path.join(DATASETS_DIR, "test.npz")
    label_json = os.path.join(DATASETS_DIR, "label_map.json")
    
    # 检查必要文件是否存在
    if not (os.path.exists(test_npz) and os.path.exists(label_json)):
        raise FileNotFoundError("缺少test.npz或label_map.json文件。请先运行build_dataset.py构建数据集。")

    # 加载测试数据
    data = np.load(test_npz)
    samples = data['samples'].astype(np.float32)  # (N, seq_len, num_channels)
    file_ids = data['file_ids']

    # 加载标签映射
    with open(label_json, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # 创建ID到类别名的反向映射
    id_to_class = {v: k for k, v in label_map.items()}

    # ==================== 模型加载 ====================
    # 查找最新的训练模型权重文件
    ckpts = [f for f in os.listdir(MODELS_DIR) if f.startswith('cnn1d_run_') and f.endswith('.pt')]
    if not ckpts:
        raise FileNotFoundError("未找到训练好的模型权重文件。请先运行train.py训练模型。")
    last_ckpt = sorted(ckpts)[-1]  # 选择最后一个模型文件

    # 从数据形状确定模型参数
    seq_len = samples.shape[1]  # 序列长度
    input_channels = samples.shape[2]  # 输入通道数

    # 创建模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(input_channels=input_channels, num_classes=len(label_map), seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, last_ckpt), map_location=device))
    model.eval()  # 设置为评估模式

    print(f"使用模型权重: {last_ckpt}")
    print(f"测试样本数量: {samples.shape[0]}")

    # ==================== 推理过程 ====================
    preds = []
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for i in range(samples.shape[0]):
            # 准备单个样本数据
            x = torch.tensor(samples[i], dtype=torch.float32).unsqueeze(0)  # (1, seq_len, num_channels)
            x = x.to(device)
            
            # 前向传播获取预测结果
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            preds.append(pred)

    # ==================== 结果聚合 ====================
    # 按文件ID聚合预测结果（同一文件的多个窗口样本）
    per_file_preds = defaultdict(list)
    for fid, p in zip(file_ids.tolist(), preds):
        per_file_preds[fid].append(p)

    # ==================== 生成提交文件 ====================
    lines = ["测试集名称\t故障类型"]  # 文件头
    for fid in sorted(per_file_preds.keys()):
        # 对同一文件的多个预测结果进行多数投票
        mv = majority_vote(np.array(per_file_preds[fid]))
        fault_class = id_to_class.get(mv, str(mv))
        # 按照要求的格式写入：文件名\t故障类型
        lines.append(f"{fid}\t{fault_class}")

    # 保存提交文件
    with open(SUBMIT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"提交文件已生成: {SUBMIT_PATH}")
    print(f"共预测了 {len(per_file_preds)} 个测试文件")


if __name__ == "__main__":
    main()