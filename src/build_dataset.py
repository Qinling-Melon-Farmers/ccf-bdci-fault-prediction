"""
数据集构建模块
负责从Excel文件中读取时序数据，通过滑动窗口切分样本，构建训练集和测试集
"""
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import TRAIN_DIR, TEST_DIR, WINDOW_SIZE, STEP_SIZE, DATASETS_DIR


def _numeric_matrix_from_excel(path: str) -> np.ndarray:
    """
    从Excel文件中提取数值矩阵
    
    Args:
        path: Excel文件路径
        
    Returns:
        np.ndarray: 形状为(seq_len, num_channels)的数值矩阵
        
    Raises:
        ValueError: 如果文件中没有数值列
    """
    df = pd.read_excel(path, engine="openpyxl")
    # 只保留数值列
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        raise ValueError(f"No numeric columns found in {path}")
    # 返回形状: (seq_len, num_channels)
    return num_df.to_numpy(dtype=np.float32)


def _extract_windows(mat: np.ndarray, window: int, step: int) -> np.ndarray:
    """
    使用滑动窗口从时序数据中提取样本
    
    Args:
        mat: 输入矩阵，形状为(seq_len, num_channels)
        window: 窗口大小
        step: 滑动步长
        
    Returns:
        np.ndarray: 窗口样本，形状为(num_windows, window, num_channels)
        
    Note:
        当step == window时，窗口之间无重叠
    """
    seq_len, num_channels = mat.shape
    windows: List[np.ndarray] = []
    
    # 滑动窗口提取
    for start in range(0, seq_len - window + 1, step):
        end = start + window
        w = mat[start:end, :]
        if w.shape[0] == window:  # 确保窗口完整
            windows.append(w)
    
    if not windows:
        return np.empty((0, window, num_channels), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def _class_name_from_filename(fname: str) -> str:
    """
    从文件名中提取类别名称
    
    Args:
        fname: 文件名
        
    Returns:
        str: 类别名称
        
    Note:
        优先使用前两个下划线分隔的部分作为类别名（如roller_broken）
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    # 优先使用前两个下划线分隔的部分作为类别（如roller_broken）
    parts = base.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else base


def build_train_dataset() -> Tuple[str, str]:
    """
    构建训练数据集
    
    解析TRAIN_DIR中的所有.xlsx文件，构建滑动窗口样本和标签映射
    保存到{DATASETS_DIR}/train.npz和{DATASETS_DIR}/label_map.json
    
    Returns:
        Tuple[str, str]: (npz文件路径, 标签映射json文件路径)
        
    Raises:
        FileNotFoundError: 如果训练目录不存在或没有找到xlsx文件
    """
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")

    all_samples: List[np.ndarray] = []
    all_labels: List[int] = []
    label_map: Dict[str, int] = {}

    # 获取所有xlsx文件（支持嵌套目录结构）
    xlsx_files = []
    for root, dirs, files in os.walk(TRAIN_DIR):
        for f in files:
            if f.lower().endswith(".xlsx"):
                xlsx_files.append(os.path.join(root, f))
    
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {TRAIN_DIR} or its subdirectories")

    # 处理每个文件
    for fpath in xlsx_files:
        # 读取数值矩阵
        mat = _numeric_matrix_from_excel(fpath)
        # 提取滑动窗口
        windows = _extract_windows(mat, WINDOW_SIZE, STEP_SIZE)
        if windows.size == 0:
            continue
        
        # 从文件路径获取类别名称（基于父目录名）
        parent_dir = os.path.basename(os.path.dirname(fpath))
        cname = _class_name_from_filename(parent_dir)
        if cname not in label_map:
            label_map[cname] = len(label_map)
        label_id = label_map[cname]
        
        # 添加样本和标签
        all_samples.append(windows)
        all_labels.append(np.full((windows.shape[0],), label_id, dtype=np.int64))

    # 合并所有样本和标签
    samples = np.concatenate(all_samples, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 保存文件
    npz_path = os.path.join(DATASETS_DIR, "train.npz")
    json_path = os.path.join(DATASETS_DIR, "label_map.json")

    np.savez(npz_path, samples=samples, labels=labels)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    return npz_path, json_path


def build_test_dataset() -> str:
    """
    构建测试数据集
    
    为每个测试文件构建滑动窗口样本，并保存文件ID信息
    保存到{DATASETS_DIR}/test.npz，包含：samples, file_ids
    
    Returns:
        str: npz文件路径
        
    Raises:
        FileNotFoundError: 如果测试目录不存在或没有找到xlsx文件
    """
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

    all_samples: List[np.ndarray] = []
    all_file_ids: List[str] = []

    # 获取所有xlsx文件并排序
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".xlsx")]
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {TEST_DIR}")

    # 处理每个测试文件
    for f in sorted(files):
        fpath = os.path.join(TEST_DIR, f)
        # 读取数值矩阵
        mat = _numeric_matrix_from_excel(fpath)
        # 提取滑动窗口
        windows = _extract_windows(mat, WINDOW_SIZE, STEP_SIZE)
        if windows.size == 0:
            continue
        
        # 获取文件ID（如test0001）
        file_id = os.path.splitext(f)[0]
        all_samples.append(windows)
        # 为每个窗口记录对应的文件ID
        all_file_ids.extend([file_id] * windows.shape[0])

    # 合并所有样本
    samples = np.concatenate(all_samples, axis=0)
    
    # 保存测试数据集
    npz_path = os.path.join(DATASETS_DIR, "test.npz")
    np.savez(npz_path, samples=samples, file_ids=np.array(all_file_ids))
    return npz_path


if __name__ == "__main__":
    # 构建训练集和测试集
    train_npz, label_json = build_train_dataset()
    test_npz = build_test_dataset()
    print(f"已保存训练集npz文件: {train_npz}")
    print(f"已保存标签映射文件: {label_json}")
    print(f"已保存测试集npz文件: {test_npz}")