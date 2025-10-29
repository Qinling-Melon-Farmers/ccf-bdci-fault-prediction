import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, mode
import glob
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import joblib
from collections import Counter

# 定义数据路径（使用脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(PROJECT_ROOT, "A", "data")
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

# 检查路径是否存在
if not os.path.exists(train_path):
    print(f"错误: 训练数据路径不存在: {train_path}")
    sys.exit(1)
if not os.path.exists(test_path):
    print(f"错误: 测试数据路径不存在: {test_path}")
    sys.exit(1)

# 定义故障类型
fault_types = {
    "inner_broken_train150": 0,  # 内圈破损
    "inner_wear_train120": 1,    # 内圈磨损
    "normal_train160": 2,        # 正常状态
    "outer_missing_train180": 3, # 外圈缺失
    "roller_broken_train150": 4, # 滚动体破损
    "roller_wear_train100": 5    # 滚动体磨损
}

# 故障类型映射（用于输出结果）
fault_type_mapping = {
    0: "inner_broken",
    1: "inner_wear",
    2: "normal",
    3: "outer_missing",
    4: "roller_broken",
    5: "roller_wear"
}

# 窗口处理参数
SAMPLE_RATE = 20480  # 采样率（假设为20.48kHz）
WINDOW_SIZE_SEC = 0.5  # 窗口大小（秒）
WINDOW_OVERLAP = 0.5  # 窗口重叠率（50%）

# 计算窗口大小和步长
WINDOW_SAMPLES = int(WINDOW_SIZE_SEC * SAMPLE_RATE)
STEP_SIZE = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))

# 阶次处理函数
def order_tracking(speed_signal, vibration_signal, sample_rate=SAMPLE_RATE):
    """
    对信号进行阶次处理，将时域信号转换为阶次域
    """
    try:
        # 检测脉冲位置（简化版，实际应根据具体信号特性调整）
        # 这里假设speed_signal中的峰值代表转速脉冲
        threshold = np.mean(speed_signal) + 2 * np.std(speed_signal)
        pulse_indices = np.where((speed_signal > threshold) & 
                                (np.roll(speed_signal, 1) <= threshold))[0]
        
        if len(pulse_indices) < 2:
            print("警告: 未检测到足够的脉冲，跳过阶次处理")
            return vibration_signal
        
        # 计算瞬时转速（RPM）
        time_indices = pulse_indices / sample_rate  # 转换为时间（秒）
        time_diffs = np.diff(time_indices)  # 相邻脉冲的时间差
        rpm_values = 60.0 / time_diffs  # 转换为RPM
        
        # 创建时间点和对应的RPM值
        rpm_times = (time_indices[:-1] + time_indices[1:]) / 2  # 取中点时间
        
        # 对所有时间点进行RPM插值
        all_times = np.arange(len(vibration_signal)) / sample_rate
        rpm_interp = interp1d(rpm_times, rpm_values, bounds_error=False, 
                             fill_value=(rpm_values[0], rpm_values[-1]))
        rpm_all = rpm_interp(all_times)
        
        # 计算累积角度（以弧度为单位）
        angle = np.cumsum(rpm_all * 2 * np.pi / 60 / sample_rate)
        
        # 创建均匀角度网格（阶次域）
        angle_min, angle_max = angle[0], angle[-1]
        uniform_angle = np.linspace(angle_min, angle_max, len(vibration_signal))
        
        # 将振动信号重采样到均匀角度网格
        vib_interp = interp1d(angle, vibration_signal, bounds_error=False, 
                             fill_value=(vibration_signal[0], vibration_signal[-1]))
        order_vib = vib_interp(uniform_angle)
        
        return order_vib
    except Exception as e:
        print(f"阶次处理出错: {e}")
        traceback.print_exc()
        return vibration_signal  # 出错时返回原始信号

# 时域特征提取
def extract_time_domain_features(signal_data):
    """提取时域特征"""
    features = []
    
    # 基本统计特征
    features.append(np.mean(signal_data))  # 均值
    features.append(np.std(signal_data))   # 标准差
    
    # 均方根值 (RMS)
    rms = np.sqrt(np.mean(np.square(signal_data)))
    features.append(rms)
    
    # 均方根偏差
    features.append(np.sqrt(np.mean(np.square(signal_data - np.mean(signal_data)))))
    
    # 偏度 (Skewness)
    features.append(skew(signal_data))
    
    # 峭度 (Kurtosis)
    features.append(kurtosis(signal_data))
    
    # 峰值 (Peak)
    peak = np.max(np.abs(signal_data))
    features.append(peak)
    
    # 谷值 (Min)
    features.append(np.min(signal_data))
    
    # 峰峰值 (Peak-to-peak)
    features.append(np.max(signal_data) - np.min(signal_data))
    
    # 峭度因子 (Crest factor = peak / RMS)
    features.append(peak / rms if rms > 0 else 0)
    
    # 脉冲指标 (Impulse factor)
    mean_abs = np.mean(np.abs(signal_data))
    features.append(peak / mean_abs if mean_abs > 0 else 0)
    
    # 裕度因子 (Clearance factor)
    features.append(peak / (np.mean(np.sqrt(np.abs(signal_data))) ** 2) if np.mean(np.sqrt(np.abs(signal_data))) > 0 else 0)
    
    # 波形因子 (Shape factor)
    features.append(rms / mean_abs if mean_abs > 0 else 0)
    
    
    return features

#包络分析特征提取
def extract_envelope_features(signal_data, fs=SAMPLE_RATE):
    """提取包络分析特征"""
    features = []
    
    # 希尔伯特变换得到解析信号与包络
    analytic_signal = signal.hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    # 包络统计特征
    features.append(np.mean(envelope))   # 包络均值
    features.append(np.std(envelope))    # 包络标准差
    features.append(np.max(envelope))    # 包络峰值
    
    # 包络RMS
    env_rms = np.sqrt(np.mean(np.square(envelope)))
    features.append(env_rms)
    
    # 包络频谱分析
    env_fft = np.abs(np.fft.fft(envelope))
    env_fft = env_fft[:len(env_fft)//2]  # 取前半部分
    
    # 包络频谱主频及其幅值
    freq_resolution = fs / len(envelope)
    frequencies = np.arange(len(env_fft)) * freq_resolution
    dominant_idx = np.argmax(env_fft)
    dominant_freq = frequencies[dominant_idx]
    dominant_amp = env_fft[dominant_idx]
    features.append(dominant_freq)       # 包络主频
    features.append(dominant_amp)        # 包络主频幅值
    
    # 包络频谱总能量
    features.append(np.sum(env_fft**2))
    
    return features



# 包络谱侧带密度特征（增强版）
def extract_envelope_sideband_features(signal_data, fs=SAMPLE_RATE):
    """提取包络谱侧带密度特征（添加阶次域侧带度量）"""
    features = []
    
    # 计算包络
    analytic_signal = signal.hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    # 包络谱
    env_fft = np.abs(np.fft.fft(envelope))
    env_fft = env_fft[:len(env_fft)//2]
    
    freq_resolution = fs / len(envelope)
    frequencies = np.arange(len(env_fft)) * freq_resolution
    
    # 估计主转频（假设在低频范围内的主峰）
    low_freq_mask = frequencies < 200  # 假设转频在200Hz以下
    if np.any(low_freq_mask):
        low_freq_spectrum = env_fft[low_freq_mask]
        main_freq_idx = np.argmax(low_freq_spectrum)
        main_freq = frequencies[low_freq_mask][main_freq_idx]
    else:
        main_freq = 50  # 默认值
    
    # 计算1x, 2x, 3x转频的侧带特征
    harmonics = [1, 2, 3]
    for h in harmonics:
        target_freq = h * main_freq
        
        # 找到目标频率附近的索引
        target_idx = int(target_freq / freq_resolution)
        
        if target_idx < len(env_fft):
            # 侧带范围（±10Hz）
            delta_f = 10
            delta_idx = int(delta_f / freq_resolution)
            
            start_idx = max(0, target_idx - delta_idx)
            end_idx = min(len(env_fft), target_idx + delta_idx)
            
            # 侧带能量占比
            sideband_energy = np.sum(env_fft[start_idx:end_idx]**2)
            total_energy = np.sum(env_fft**2)
            
            if total_energy > 0:
                energy_ratio = sideband_energy / total_energy
                features.append(energy_ratio)
            else:
                features.append(0)
            
            # 侧带计数（超过阈值的峰数量）
            threshold = np.mean(env_fft[start_idx:end_idx]) + np.std(env_fft[start_idx:end_idx])
            sideband_count = np.sum(env_fft[start_idx:end_idx] > threshold)
            features.append(sideband_count)
        else:
            features.extend([0, 0])
    
    # 新增：阶次域侧带度量特征
    try:
        # 轴承故障特征频率（相对于转频的倍数）
        bearing_orders = {
            'BPFI': 7.5,    # 内圈故障频率（示例值）
            'BPFO': 4.5,    # 外圈故障频率（示例值）
            'BSF': 2.8,     # 滚动体故障频率（示例值）
            'FTF': 0.4      # 保持架故障频率（示例值）
        }
        
        for fault_name, order in bearing_orders.items():
            # 计算故障特征频率
            fault_freq = order * main_freq
            fault_idx = int(fault_freq / freq_resolution)
            
            if fault_idx < len(env_fft):
                # 故障频率及其侧带的能量分析
                sideband_width = int(5 / freq_resolution)  # ±5Hz侧带宽度
                
                start_idx = max(0, fault_idx - sideband_width)
                end_idx = min(len(env_fft), fault_idx + sideband_width)
                
                # 故障频率峰值
                fault_peak = env_fft[fault_idx]
                features.append(fault_peak)
                
                # 故障频率侧带能量
                fault_sideband_energy = np.sum(env_fft[start_idx:end_idx]**2)
                features.append(fault_sideband_energy)
                
                # 故障频率侧带密度（峰值数量）
                local_mean = np.mean(env_fft[start_idx:end_idx])
                local_std = np.std(env_fft[start_idx:end_idx])
                threshold = local_mean + 2 * local_std
                sideband_peaks = np.sum(env_fft[start_idx:end_idx] > threshold)
                features.append(sideband_peaks)
                
                # 故障频率调制深度（侧带与载波的比值）
                if fault_peak > 0:
                    modulation_depth = fault_sideband_energy / (fault_peak**2)
                    features.append(modulation_depth)
                else:
                    features.append(0)
            else:
                # 如果故障频率超出范围，添加默认值
                features.extend([0, 0, 0, 0])
        
        # 总体侧带特征
        # 计算所有故障频率的综合侧带指标
        all_fault_freqs = [order * main_freq for order in bearing_orders.values()]
        valid_fault_indices = [int(f / freq_resolution) for f in all_fault_freqs 
                              if int(f / freq_resolution) < len(env_fft)]
        
        if valid_fault_indices:
            # 所有故障频率的平均峰值
            avg_fault_peak = np.mean([env_fft[idx] for idx in valid_fault_indices])
            features.append(avg_fault_peak)
            
            # 故障频率与背景噪声的对比度
            background_indices = [i for i in range(len(env_fft)) if i not in valid_fault_indices]
            if background_indices:
                background_level = np.mean([env_fft[i] for i in background_indices[:100]])  # 取前100个背景点
                if background_level > 0:
                    contrast_ratio = avg_fault_peak / background_level
                    features.append(contrast_ratio)
                else:
                    features.append(0)
            else:
                features.append(0)
        else:
            features.extend([0, 0])
            
    except Exception as e:
        print(f"阶次域侧带特征提取出错: {e}")
        # 如果出错，添加默认值（4个故障类型 × 4个特征 + 2个总体特征）
        features.extend([0] * 18)
    
    return features

# 频域特征提取（肖怀瑾代码中的）
def extract_frequency_domain_features(signal_data, fs=SAMPLE_RATE):
    """提取频域特征"""
    features = []
    
    # 计算FFT
    fft_result = np.abs(np.fft.fft(signal_data))
    # 取前半部分（由于对称性）
    fft_result = fft_result[:len(fft_result)//2]
    
    # 频率分辨率
    freq_resolution = fs / len(signal_data)
    frequencies = np.arange(len(fft_result)) * freq_resolution
    
    # 频域基本特征
    features.append(np.mean(fft_result))  # 频域均值
    features.append(np.std(fft_result))   # 频域标准差
    features.append(np.max(fft_result))   # 频域峰值
    
    # 频域总能量
    total_energy = np.sum(fft_result**2)
    features.append(total_energy)
    
    # 主频及其幅值
    dominant_idx = np.argmax(fft_result)
    dominant_freq = frequencies[dominant_idx]
    dominant_amp = fft_result[dominant_idx]
    features.append(dominant_freq)  # 主频
    features.append(dominant_amp)   # 主频幅值
    
    # 频谱质心 (Spectral centroid)
    if np.sum(fft_result) > 0:
        spectral_centroid = np.sum(frequencies * fft_result) / np.sum(fft_result)
        features.append(spectral_centroid)
    else:
        features.append(0)
    
    # 频谱熵 (Spectral entropy)
    normalized_fft = fft_result / np.sum(fft_result) if np.sum(fft_result) > 0 else np.zeros_like(fft_result)
    spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-10))
    features.append(spectral_entropy)
    
    # 分段频带能量
    # 将频谱分为5个频带，计算每个频带的能量
    band_size = len(fft_result) // 5
    for i in range(5):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < 4 else len(fft_result)
        band_energy = np.sum(fft_result[start_idx:end_idx]**2)
        features.append(band_energy / total_energy if total_energy > 0 else 0)  # 归一化的频带能量
    
    # 前N个谐波幅值比
    N = 5  # 取前5个谐波
    if dominant_idx > 0:
        for i in range(2, N+2):  # 从2倍频开始
            harmonic_idx = min(int(dominant_idx * i), len(fft_result)-1)
            harmonic_amp = fft_result[harmonic_idx]
            features.append(harmonic_amp / dominant_amp if dominant_amp > 0 else 0)  # 谐波幅值比
    else:
        features.extend([0] * N)  # 如果没有检测到主频，填充0
    
    return features

# 小波特征提取（增强版）
def extract_wavelet_features(signal_data):
    """提取小波特征（增强版）"""
    features = []
    
    # 使用小波变换的多尺度分解
    # 定义几个尺度级别（优化后的尺度）
    scales = [2, 4, 8, 16, 32]
    
    # 原始多尺度能量特征
    scale_energies = []
    for scale in scales:
        # 使用简单的均值滤波模拟小波分解的不同尺度
        kernel = np.ones(scale) / scale
        # 对信号进行卷积
        filtered = np.convolve(signal_data, kernel, mode='same')
        # 计算该尺度下的能量
        energy = np.sum(filtered**2)
        scale_energies.append(energy)
        features.append(energy)
    
    # 计算各尺度能量的分布
    total_energy = np.sum(scale_energies)
    if total_energy > 0:
        normalized_energies = [e / total_energy for e in scale_energies]
        features.extend(normalized_energies)  # 添加归一化的能量分布
    else:
        features.extend([0] * len(scales))
    
    # 新增：尺度间的能量比值特征
    if total_energy > 0:
        # 低频与高频能量比
        low_freq_energy = sum(scale_energies[:2])  # 尺度2,4
        high_freq_energy = sum(scale_energies[3:])  # 尺度16,32
        if high_freq_energy > 0:
            features.append(low_freq_energy / high_freq_energy)
        else:
            features.append(0)
        
        # 相邻尺度能量比
        for i in range(len(scale_energies) - 1):
            if scale_energies[i+1] > 0:
                features.append(scale_energies[i] / scale_energies[i+1])
            else:
                features.append(0)
    else:
        features.extend([0] * 5)  # 1个低高频比 + 4个相邻尺度比
    
    # 新增：多尺度统计特征
    try:
        # 计算每个尺度下的统计特征
        for scale in scales[:3]:  # 只取前3个尺度以减少计算量
            kernel = np.ones(scale) / scale
            filtered = np.convolve(signal_data, kernel, mode='same')
            
            # 该尺度下的统计特征
            if len(filtered) > 3 and np.std(filtered) > 1e-10:
                features.append(np.std(filtered))      # 标准差
                features.append(skew(filtered))        # 偏度
                features.append(kurtosis(filtered))    # 峭度
            else:
                features.extend([0, 0, 0])
    except Exception as e:
        print(f"多尺度统计特征计算出错: {e}")
        features.extend([0] * 9)  # 3个尺度 × 3个统计量
    
    # 新增：小波能量熵
    if total_energy > 0:
        # 计算能量熵
        energy_probs = [e / total_energy for e in scale_energies if e > 0]
        if len(energy_probs) > 1:
            wavelet_entropy = -sum(p * np.log2(p) for p in energy_probs)
            features.append(wavelet_entropy)
        else:
            features.append(0)
    else:
        features.append(0)
    
    return features


# 窗口化处理函数
def apply_windowing(signal_data, window_size=WINDOW_SAMPLES, step_size=STEP_SIZE):
    """
    对信号进行窗口化处理，返回窗口列表
    """
    windows = []
    for i in range(0, len(signal_data) - window_size + 1, step_size):
        window = signal_data[i:i + window_size]
        windows.append(window)
    return windows

# 特征提取函数
def extract_features(file_path, use_order_tracking=True, use_windowing=True):
    """从Excel文件中提取特征"""
    try:
        print(f"正在处理文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return None
            
        # 读取Excel文件
        try:
            # 尝试使用openpyxl引擎
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e1:
            print(f"使用openpyxl引擎读取失败: {e1}")
            try:
                # 尝试使用xlrd引擎
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e2:
                print(f"使用xlrd引擎读取失败: {e2}")
                raise Exception(f"无法读取Excel文件: {e1}, {e2}")
        
        # 打印数据形状和前几行，用于调试
        print(f"数据形状: {df.shape}")
        
        # 获取转速脉冲信号和垂直方向振动信号
        if df.shape[1] < 2:
            raise Exception(f"数据列数不足，需要至少2列，但只有{df.shape[1]}列")
            
        speed_signal = df.iloc[:, 0].values
        vibration_signal = df.iloc[:, 1].values
        
        # 检查数据是否为空或包含NaN
        if len(speed_signal) == 0 or len(vibration_signal) == 0:
            raise Exception("数据为空")
        if np.isnan(speed_signal).any() or np.isnan(vibration_signal).any():
            raise Exception("数据包含NaN值")
        
        # 阶次处理（如果启用）
        if use_order_tracking:
            vibration_signal = order_tracking(speed_signal, vibration_signal)
        
        # 窗口化处理（如果启用）
        if use_windowing:
            windows = apply_windowing(vibration_signal)
            if not windows:
                print("警告: 窗口化处理后没有窗口，使用整个信号")
                windows = [vibration_signal]
        else:
            windows = [vibration_signal]
        
        # 对每个窗口提取特征，然后取平均值
        all_features = []
        
        for window in windows:
            window_features = []
            
            # 提取时域特征
            time_features = extract_time_domain_features(window)
            window_features.extend(time_features)
            
            # 提取频域特征
            freq_features = extract_frequency_domain_features(window)
            window_features.extend(freq_features)
            
            # 提取包络特征
            envelope_features = extract_envelope_features(window)
            window_features.extend(envelope_features)
            
            # 提取小波特征
            wavelet_features = extract_wavelet_features(window)
            window_features.extend(wavelet_features)

            #提取包络谱侧带密度特征（增强版）
            envelope_sideband_features = extract_envelope_sideband_features(windows)
            window_features.extend(envelope_sideband_features)
            
            all_features.append(window_features)
        
        # 计算所有窗口特征的平均值
        avg_features = np.mean(all_features, axis=0)
        
        return avg_features
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        traceback.print_exc()  # 打印完整的错误堆栈
        return None

# 加载训练数据
def load_training_data(use_order_tracking=True, use_windowing=True):
    """加载训练数据并提取特征"""
    X = []
    y = []
    for fault_type, label in fault_types.items():
        fault_dir = os.path.join(train_path, fault_type)
        files = glob.glob(os.path.join(fault_dir, "*.xlsx"))
        print(f"正在处理 {fault_type} 类型的数据，共 {len(files)} 个文件")
        for file in files:
            features = extract_features(file, use_order_tracking, use_windowing)
            if features is not None:
                X.append(features)
                y.append(label)
    # 将列表转为数组
    X_array = np.array(X, dtype=float)
    # 非空校验：避免后续标准化接收空数组
    if X_array.size == 0 or X_array.shape[0] == 0:
        raise ValueError("load_training_data: 未收集到任何有效特征（X为空）。请检查数据文件与特征提取流程。")
    # 保证二维形状
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    print(f"特征数组形状: {X_array.shape}，样本数: {X_array.shape[0]}")
    return X_array, np.array(y)

# 特征选择和降维
def feature_selection(X, y, variance_threshold=0.01, n_components=None):
    """特征选择和降维"""
    # 方差过滤
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var_filtered = selector.fit_transform(X)
    
    print(f"方差过滤后特征数: {X_var_filtered.shape[1]}")
    
    # 如果指定了PCA组件数，则进行PCA降维
    if n_components is not None and n_components < X_var_filtered.shape[1]:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_var_filtered)
        print(f"PCA降维后特征数: {X_reduced.shape[1]}")
        print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")
        return X_reduced, selector, pca
    
    return X_var_filtered, selector, None

# 训练随机森林模型
def train_rf_model(X, y):
    """训练随机森林分类器"""
    # 非空校验
    if X is None or np.asarray(X).shape[0] == 0:
        raise ValueError("train_rf_model: 训练特征为空（X.shape[0]==0）。请检查前序数据收集。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"随机森林验证集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("随机森林特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca, accuracy, f1

# 训练XGBoost模型
def train_xgb_model(X, y):
    """训练XGBoost分类器"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练XGBoost模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(fault_types),
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"XGBoost验证集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("XGBoost特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca, accuracy, f1

# 训练SVM模型
def train_svm_model(X, y):
    """训练SVM分类器"""
    # 非空校验
    if X is None or np.asarray(X).shape[0] == 0:
        raise ValueError("train_svm_model: 训练特征为空（X.shape[0]==0）。请检查前序数据收集。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练SVM模型
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"SVM验证集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    return model, scaler, selector, pca, accuracy, f1

# 训练投票集成模型
def train_voting_ensemble(X, y, rf_model, xgb_model, svm_model):
    """训练投票集成模型"""
    # 非空校验
    if X is None or np.asarray(X).shape[0] == 0:
        raise ValueError("train_voting_ensemble: 训练特征为空（X.shape[0]==0）。请检查前序数据收集。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 创建投票集成模型
    voting_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svm', svm_model)
        ],
        voting='soft',  # 使用软投票（基于概率）
        weights=None  # 可以设置为基于验证集性能的权重
    )
    
    # 训练投票集成模型
    voting_model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = voting_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"投票集成验证集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    return voting_model, scaler, selector, pca, accuracy, f1

# 训练Stacking集成模型
def train_stacking_ensemble(X, y):
    """训练Stacking集成模型"""
    # 非空校验
    if X is None or np.asarray(X).shape[0] == 0:
        raise ValueError("train_stacking_ensemble: 训练特征为空（X.shape[0]==0）。请检查前序数据收集。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 定义基学习器
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(fault_types),
            random_state=42
        )),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ]
    
    # 定义元学习器
    meta_learner = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000)
    
    # 创建Stacking集成模型
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,  # 5折交叉验证
        passthrough=False,  # 是否使用原始特征
        stack_method='auto'
    )
    
    # 训练Stacking集成模型
    stacking_model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = stacking_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"Stacking集成验证集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
    print(classification_report(y_val, y_pred))
    
    return stacking_model, scaler, selector, pca, accuracy, f1

# 预测测试数据
def predict_test_data(model, scaler, selector=None, pca=None, use_order_tracking=True, use_windowing=True, model_name="Model"):
    """预测测试数据集中的故障类型
    
    参数:
        model: 训练好的模型
        scaler: 标准化器
        selector: 特征选择器 (可选)
        pca: PCA降维器 (可选)
        use_order_tracking: 是否使用阶次处理
        use_windowing: 是否使用窗口处理
        model_name: 模型名称
    """
    test_files = glob.glob(os.path.join(test_path, "*.xlsx"))
    results = []
    probabilities = []
    
    print(f"使用{model_name}模型预测测试数据...")
    
    for file in test_files:
        file_name = os.path.basename(file)
        features = extract_features(file, use_order_tracking, use_windowing)
        
        if features is not None:
            # 标准化特征
            features_scaled = scaler.transform([features])
            
            # 应用特征选择
            if selector is not None:
                features_scaled = selector.transform(features_scaled)
            
            # 应用PCA降维
            if pca is not None:
                features_scaled = pca.transform(features_scaled)
                
            # 预测故障类型
            prediction = model.predict(features_scaled)[0]
            # 获取故障类型名称
            fault_name = list(fault_types.keys())[list(fault_types.values()).index(prediction)]
            # 去掉后缀，只保留故障类型
            fault_name = fault_name.split('_train')[0]
            
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                probabilities.append((file_name, proba))
            else:
                probabilities.append((file_name, None))
            
            results.append((file_name, fault_name))
            print(f"文件: {file_name}, 预测故障类型: {fault_name}")
    
    # 按文件名排序
    results.sort(key=lambda x: x[0])
    probabilities.sort(key=lambda x: x[0])
    
    return results, probabilities

# 保存预测结果
def save_results(results, output_file="prediction_results.txt"):
    """保存预测结果到文件"""
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    with open(output_file, 'w') as f:
        for file_name, fault_type in results:
            f.write(f"{file_name},{fault_type}\n")
    print(f"预测结果已保存到 {output_file}")

# 基于概率的集成方法
def ensemble_by_probability(rf_probs, xgb_probs, svm_probs, weights=None):
    """基于概率的集成预测
    
    参数:
        rf_probs: 随机森林的概率预测结果
        xgb_probs: XGBoost的概率预测结果  
        svm_probs: SVM的概率预测结果
        weights: 各模型的权重，如果不指定则使用等权重
    
    返回:
        集成后的预测结果
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]  # 等权重
    
    ensemble_results = []
    
    # 确保所有模型都有相同数量的预测结果
    file_names = [item[0] for item in rf_probs]
    
    for i, file_name in enumerate(file_names):
        # 获取各模型的概率
        rf_prob = rf_probs[i][1]
        xgb_prob = xgb_probs[i][1]
        svm_prob = svm_probs[i][1]
        
        # 计算加权平均概率
        if rf_prob is not None and xgb_prob is not None and svm_prob is not None:
            avg_prob = (weights[0] * rf_prob + weights[1] * xgb_prob + weights[2] * svm_prob) / sum(weights)
            final_pred = np.argmax(avg_prob)
        else:
            # 如果有模型没有概率输出，使用简单投票
            predictions = []
            if rf_prob is not None:
                predictions.append(np.argmax(rf_prob))
            if xgb_prob is not None:
                predictions.append(np.argmax(xgb_prob))
            if svm_prob is not None:
                predictions.append(np.argmax(svm_prob))
            
            if predictions:
                final_pred = mode(predictions)[0][0]
            else:
                final_pred = 0  # 默认值
        
        # 获取故障类型名称
        fault_name = list(fault_types.keys())[list(fault_types.values()).index(final_pred)]
        fault_name = fault_name.split('_train')[0]
        
        ensemble_results.append((file_name, fault_name))
    
    return ensemble_results

# 基于性能加权的集成方法
def ensemble_by_performance_weight(rf_results, xgb_results, svm_results, rf_acc, xgb_acc, svm_acc):
    """基于验证集性能加权的集成预测
    
    参数:
        rf_results: 随机森林的预测结果
        xgb_results: XGBoost的预测结果
        svm_results: SVM的预测结果
        rf_acc: 随机森林的验证集准确率
        xgb_acc: XGBoost的验证集准确率
        svm_acc: SVM的验证集准确率
    
    返回:
        集成后的预测结果
    """
    # 计算权重（基于验证集准确率）
    total_acc = rf_acc + xgb_acc + svm_acc
    weights = [rf_acc / total_acc, xgb_acc / total_acc, svm_acc / total_acc]
    
    print(f"模型权重 - RF: {weights[0]:.4f}, XGB: {weights[1]:.4f}, SVM: {weights[2]:.4f}")
    
    ensemble_results = []
    
    # 创建文件名到预测标签的映射
    rf_dict = {file: fault for file, fault in rf_results}
    xgb_dict = {file: fault for file, fault in xgb_results}
    svm_dict = {file: fault for file, fault in svm_results}
    
    # 创建故障类型到数值的映射
    fault_to_num = {fault: idx for idx, fault in enumerate(fault_types.keys())}
    
    # 获取所有文件名
    all_files = sorted(list(set(list(rf_dict.keys()) + list(xgb_dict.keys()) + list(svm_dict.keys()))))
    
    for file in all_files:
        # 获取各模型的预测
        rf_pred = rf_dict.get(file, None)
        xgb_pred = xgb_dict.get(file, None)
        svm_pred = svm_dict.get(file, None)
        
        # 转换为数值
        predictions = []
        pred_weights = []
        
        if rf_pred:
            rf_num = fault_to_num.get(rf_pred + "_train150", 0)  # 使用任意后缀
            predictions.append(rf_num)
            pred_weights.append(weights[0])
        
        if xgb_pred:
            xgb_num = fault_to_num.get(xgb_pred + "_train150", 0)
            predictions.append(xgb_num)
            pred_weights.append(weights[1])
        
        if svm_pred:
            svm_num = fault_to_num.get(svm_pred + "_train150", 0)
            predictions.append(svm_num)
            pred_weights.append(weights[2])
        
        if predictions:
            # 计算加权投票
            weighted_votes = {}
            for pred, weight in zip(predictions, pred_weights):
                if pred in weighted_votes:
                    weighted_votes[pred] += weight
                else:
                    weighted_votes[pred] = weight
            
            # 选择权重最大的预测
            final_pred = max(weighted_votes, key=weighted_votes.get)
        else:
            final_pred = 0  # 默认值
        
        # 获取故障类型名称
        fault_name = list(fault_types.keys())[final_pred]
        fault_name = fault_name.split('_train')[0]
        
        ensemble_results.append((file, fault_name))
        
        # 打印不一致的预测
        if rf_pred != xgb_pred or rf_pred != svm_pred or xgb_pred != svm_pred:
            print(f"文件: {file}, 预测不一致 - RF: {rf_pred}, XGB: {xgb_pred}, SVM: {svm_pred}, 最终: {fault_name}")
    
    return ensemble_results

# 保存提交格式的结果
def save_results_for_submission(results, output_file="submit_result.txt"):
    """按提交格式保存预测结果
    
    格式：首行是"测试集名称\t故障类型"，其后每行如"test0001\tinner_wear"。
    参数：
        results: 列表 [(file_name, fault_type)], file_name通常为"testXXXX.xlsx"
        output_file: 输出文件名，若为相对路径则写入到PROJECT_ROOT
    """
    # 解析与整理行
    rows = []
    for file_name, fault_type in results:
        name_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        # 尝试提取数字用于排序
        try:
            num = int(''.join([c for c in name_no_ext if c.isdigit()]))
        except Exception:
            num = 0
        rows.append((name_no_ext, fault_type if fault_type else "unknown", num))
    # 按编号排序
    rows.sort(key=lambda x: x[2])
    # 解析输出路径到项目根目录
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    # 写文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("测试集名称\t故障类型\n")
        for name_no_ext, fault_type, _ in rows:
            f.write(f"{name_no_ext}\t{fault_type}\n")
    print(f"提交结果已保存到 {output_file}")

# 模型评估和选择
def evaluate_models(X, y):
    """评估所有模型的性能"""
    print("正在进行模型评估...")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 定义模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(fault_types),
            random_state=42
        ),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }
    
    # 交叉验证评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"评估 {name} 模型...")
        cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'scores': cv_scores
        }
        print(f"{name} 交叉验证准确率: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    return results

# 主函数
def main():
    """主函数，执行完整的故障预测流程"""
    print("开始故障预测任务...")
    
    # 设置参数
    use_order_tracking = True  # 是否使用阶次处理
    use_windowing = True       # 是否使用窗口处理
    
    print(f"参数设置: 阶次处理={use_order_tracking}, 窗口处理={use_windowing}")
    
    # 加载训练数据
    print("正在加载训练数据...")
    X, y = load_training_data(use_order_tracking, use_windowing)
    
    # 评估所有模型的性能
    model_results = evaluate_models(X, y)
    
    # 训练各个模型
    print("\n训练随机森林模型...")
    rf_model, rf_scaler, rf_selector, rf_pca, rf_acc, rf_f1 = train_rf_model(X, y)
    
    print("\n训练XGBoost模型...")
    xgb_model, xgb_scaler, xgb_selector, xgb_pca, xgb_acc, xgb_f1 = train_xgb_model(X, y)
    
    print("\n训练SVM模型...")
    svm_model, svm_scaler, svm_selector, svm_pca, svm_acc, svm_f1 = train_svm_model(X, y)
    
    # 训练集成模型
    print("\n训练投票集成模型...")
    voting_model, voting_scaler, voting_selector, voting_pca, voting_acc, voting_f1 = train_voting_ensemble(X, y, rf_model, xgb_model, svm_model)
    
    print("\n训练Stacking集成模型...")
    stacking_model, stacking_scaler, stacking_selector, stacking_pca, stacking_acc, stacking_f1 = train_stacking_ensemble(X, y)
    
    # 使用各个模型预测测试数据
    print("\n使用随机森林模型预测测试数据...")
    rf_results, rf_probs = predict_test_data(rf_model, rf_scaler, rf_selector, rf_pca, 
                                           use_order_tracking, use_windowing, "RandomForest")
    
    print("\n使用XGBoost模型预测测试数据...")
    xgb_results, xgb_probs = predict_test_data(xgb_model, xgb_scaler, xgb_selector, xgb_pca, 
                                              use_order_tracking, use_windowing, "XGBoost")
    
    print("\n使用SVM模型预测测试数据...")
    svm_results, svm_probs = predict_test_data(svm_model, svm_scaler, svm_selector, svm_pca, 
                                              use_order_tracking, use_windowing, "SVM")
    
    print("\n使用投票集成模型预测测试数据...")
    voting_results, voting_probs = predict_test_data(voting_model, voting_scaler, voting_selector, voting_pca, 
                                                    use_order_tracking, use_windowing, "Voting Ensemble")
    
    print("\n使用Stacking集成模型预测测试数据...")
    stacking_results, stacking_probs = predict_test_data(stacking_model, stacking_scaler, stacking_selector, stacking_pca, 
                                                        use_order_tracking, use_windowing, "Stacking Ensemble")
    
    # 保存各个模型的预测结果
    save_results(rf_results, "rf_prediction_results.txt")
    save_results(xgb_results, "xgb_prediction_results.txt")
    save_results(svm_results, "svm_prediction_results.txt")
    save_results(voting_results, "voting_prediction_results.txt")
    save_results(stacking_results, "stacking_prediction_results.txt")
    
    # 多种集成策略
    print("\n应用多种集成策略...")
    
    # 1. 基于概率的集成
    print("1. 基于概率的集成...")
    proba_ensemble_results = ensemble_by_probability(rf_probs, xgb_probs, svm_probs)
    save_results(proba_ensemble_results, "proba_ensemble_results.txt")
    save_results_for_submission(proba_ensemble_results, "submit_proba_ensemble.txt")
    
    # 2. 基于性能加权的集成
    print("2. 基于性能加权的集成...")
    weighted_ensemble_results = ensemble_by_performance_weight(rf_results, xgb_results, svm_results, rf_acc, xgb_acc, svm_acc)
    save_results(weighted_ensemble_results, "weighted_ensemble_results.txt")
    save_results_for_submission(weighted_ensemble_results, "submit_weighted_ensemble.txt")
    
    # 3. 选择最佳单一模型
    print("3. 选择最佳单一模型...")
    model_accuracies = {
        'RandomForest': rf_acc,
        'XGBoost': xgb_acc,
        'SVM': svm_acc,
        'Voting': voting_acc,
        'Stacking': stacking_acc
    }
    
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    print(f"最佳模型: {best_model_name}, 准确率: {model_accuracies[best_model_name]:.4f}")
    
    if best_model_name == 'RandomForest':
        best_results = rf_results
    elif best_model_name == 'XGBoost':
        best_results = xgb_results
    elif best_model_name == 'SVM':
        best_results = svm_results
    elif best_model_name == 'Voting':
        best_results = voting_results
    else:
        best_results = stacking_results
    
    save_results(best_results, "best_single_model_results.txt")
    save_results_for_submission(best_results, "submit_best_single_model.txt")
    
    # 4. 最终集成（结合所有策略）
    print("4. 最终集成策略...")
    # 使用Stacking集成作为最终结果（通常性能最好）
    final_results = stacking_results
    save_results(final_results, "final_ensemble_results.txt")
    save_results_for_submission(final_results, "submit_final_ensemble.txt")
    
    # 打印模型性能总结
    print("\n" + "="*50)
    print("模型性能总结:")
    print("="*50)
    for model_name, acc in model_accuracies.items():
        print(f"{model_name}: {acc:.4f}")
    
    print(f"\n最终选择的模型: Stacking Ensemble")
    print(f"最终准确率: {stacking_acc:.4f}")
    
    print("\n故障预测任务完成!")

# 程序入口
if __name__ == "__main__":
    main()