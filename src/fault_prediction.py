import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, log_loss, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import glob
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

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

# 增强脉冲检测函数
def enhanced_pulse_detection(speed_signal, sample_rate=SAMPLE_RATE):
    """
    增强的脉冲检测，使用多级阈值和小波变换
    """
    try:
        # 方法1: 多级阈值检测
        thr1 = np.mean(speed_signal) + 1.2 * np.std(speed_signal)  # 降低阈值
        thr2 = np.mean(speed_signal) + 1.0 * np.std(speed_signal)  # 更低阈值
        min_dist = int(0.02 * sample_rate)
        
        # 首先尝试较低阈值
        peaks1, _ = find_peaks(speed_signal, height=thr1, 
                              prominence=0.8*np.std(speed_signal), 
                              distance=min_dist)
        
        # 如果检测到的脉冲不足，使用更低阈值
        if len(peaks1) < 3:
            peaks1, _ = find_peaks(speed_signal, height=thr2, 
                                  prominence=0.6*np.std(speed_signal), 
                                  distance=min_dist)
        
        # 方法2: 小波变换增强检测
        try:
            from scipy import signal as scipy_signal
            # 使用墨西哥帽小波进行脉冲检测
            widths = np.arange(1, min(31, len(speed_signal)//10))
            cwt_matrix = scipy_signal.cwt(speed_signal, scipy_signal.ricker, widths)
            # 取最大响应的尺度
            cwt_peaks = np.argmax(np.abs(cwt_matrix), axis=0)
            
            # 在小波变换结果上检测峰值
            cwt_signal = np.max(np.abs(cwt_matrix), axis=0)
            cwt_thr = np.mean(cwt_signal) + 0.8 * np.std(cwt_signal)
            peaks2, _ = find_peaks(cwt_signal, height=cwt_thr, distance=min_dist)
            
            # 合并两种方法的结果
            all_peaks = np.unique(np.concatenate([peaks1, peaks2]))
            
        except ImportError:
            # 如果scipy不可用，使用传统方法
            all_peaks = peaks1
        
        return all_peaks
        
    except Exception as e:
        print(f"增强脉冲检测出错: {e}")
        # 回退到基本检测
        thr = np.mean(speed_signal) + 1.5 * np.std(speed_signal)
        min_dist = int(0.02 * sample_rate)
        peaks, _ = find_peaks(speed_signal, height=thr, distance=min_dist)
        return peaks

# 阶次处理函数
def order_tracking(speed_signal, vibration_signal, sample_rate=SAMPLE_RATE):
    """
    对信号进行阶次处理，将时域信号转换为阶次域
    增强了脉冲检测和插值方法
    """
    try:
        # 使用增强的脉冲检测
        pulse_indices = enhanced_pulse_detection(speed_signal, sample_rate)
        
        if len(pulse_indices) < 2:
            print("警告: 未检测到足够的脉冲，跳过阶次处理")
            return vibration_signal
        
        # 计算瞬时转速（RPM）
        time_indices = pulse_indices / sample_rate  # 转换为时间（秒）
        time_diffs = np.diff(time_indices)  # 相邻脉冲的时间差
        rpm_values = 60.0 / time_diffs  # 转换为RPM
        
        # 创建时间点和对应的RPM值
        rpm_times = (time_indices[:-1] + time_indices[1:]) / 2  # 取中点时间
        
        # 对所有时间点进行RPM插值 - 使用样条插值
        all_times = np.arange(len(vibration_signal)) / sample_rate
        
        # 使用样条插值替代线性插值，提供更平滑的结果
        try:
            from scipy.interpolate import CubicSpline
            # 样条插值
            spline = CubicSpline(rpm_times, rpm_values, bc_type='natural')
            rpm_all = spline(all_times)
            
            # 确保RPM值在合理范围内
            rpm_all = np.clip(rpm_all, np.min(rpm_values)*0.5, np.max(rpm_values)*1.5)
            
        except ImportError:
            # 如果scipy不可用，回退到线性插值
            rpm_interp = interp1d(rpm_times, rpm_values, bounds_error=False, 
                                 fill_value=(rpm_values[0], rpm_values[-1]))
            rpm_all = rpm_interp(all_times)
        
        # 计算累积角度（以弧度为单位）
        angle = np.cumsum(rpm_all * 2 * np.pi / 60 / sample_rate)
        
        # 创建均匀角度网格（阶次域）
        angle_min, angle_max = angle[0], angle[-1]
        uniform_angle = np.linspace(angle_min, angle_max, len(vibration_signal))
        
        # 将振动信号重采样到均匀角度网格 - 也使用样条插值
        try:
            from scipy.interpolate import CubicSpline
            vib_spline = CubicSpline(angle, vibration_signal, bc_type='natural')
            order_vib = vib_spline(uniform_angle)
        except ImportError:
            # 回退到线性插值
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

# 轻量级Kurtogram频带选择（优化版）
def lightweight_kurtogram_band_selection(signal_data, fs=SAMPLE_RATE, n_levels=3, n_bands=6):
    """轻量级Kurtogram最佳频带选择（减少计算复杂度）
    
    参数:
        signal_data: 输入信号
        fs: 采样频率
        n_levels: 分解层数（减少到3层）
        n_bands: 每层的频带数（减少到6个）
    
    返回:
        best_band_signal: 最佳频带滤波后的信号
        best_kurtosis: 最佳频带的峭度值
        best_freq_range: 最佳频带的频率范围
    """
    try:
        # 如果信号太短，直接返回原信号
        if len(signal_data) < 100:
            return signal_data, kurtosis(signal_data, bias=False), (0, fs/2)
        
        # 初始化最佳参数
        best_kurtosis = -np.inf
        best_band_signal = signal_data
        best_freq_range = (0, fs/2)
        
        # 对每个分解层进行处理（减少层数以提高效率）
        for level in range(1, n_levels + 1):
            # 计算当前层的频带宽度
            band_width = fs / (2 ** (level + 1))
            
            # 跳过太窄的频带
            if band_width < 10:  # 频带宽度小于10Hz时跳过
                continue
            
            # 对每个频带进行处理（减少频带数）
            for band_idx in range(n_bands):
                # 计算频带的中心频率和范围
                center_freq = (band_idx + 0.5) * band_width
                low_freq = max(10, center_freq - band_width/2)  # 最低频率设为10Hz
                high_freq = min(fs/2 - 10, center_freq + band_width/2)  # 最高频率留10Hz余量
                
                # 跳过无效频带
                if high_freq <= low_freq or high_freq - low_freq < 20:
                    continue
                
                # 设计带通滤波器
                try:
                    # 归一化频率
                    nyquist = fs / 2
                    low_norm = low_freq / nyquist
                    high_norm = high_freq / nyquist
                    
                    # 确保频率在有效范围内
                    low_norm = max(0.02, min(0.98, low_norm))
                    high_norm = max(0.02, min(0.98, high_norm))
                    
                    if high_norm <= low_norm or high_norm - low_norm < 0.01:
                        continue
                    
                    # 使用更简单的滤波器设计
                    sos = signal.butter(2, [low_norm, high_norm], btype='band', output='sos')
                    
                    # 应用滤波器
                    filtered_signal = signal.sosfilt(sos, signal_data)
                    
                    # 计算滤波后信号的峭度
                    if len(filtered_signal) > 10 and np.std(filtered_signal) > 1e-8:
                        band_kurtosis = kurtosis(filtered_signal, bias=False)
                        
                        # 检查峭度是否有效
                        if not np.isnan(band_kurtosis) and not np.isinf(band_kurtosis):
                            # 更新最佳频带
                            if band_kurtosis > best_kurtosis:
                                best_kurtosis = band_kurtosis
                                best_band_signal = filtered_signal
                                best_freq_range = (low_freq, high_freq)
                    
                except Exception as e:
                    # 滤波器设计失败，跳过此频带
                    continue
        
        return best_band_signal, best_kurtosis, best_freq_range
    
    except Exception as e:
        print(f"Kurtogram频带选择出错: {e}")
        return signal_data, kurtosis(signal_data, bias=False), (0, fs/2)

# 谱峭度特征提取（优化版）
def extract_spectral_kurtosis_features(signal_data, fs=SAMPLE_RATE):
    """提取谱峭度特征（添加Kurtogram最佳频带选择）"""
    features = []
    
    # 原有的STFT谱峭度计算
    f, t, Zxx = signal.stft(signal_data, fs=fs, nperseg=256, noverlap=128)
    psd = np.abs(Zxx)**2
    
    # 计算每个频率的峭度
    sk_values = []
    for i in range(len(f)):
        freq_series = psd[i, :]
        if len(freq_series) > 3 and np.std(freq_series) > 0:
            sk = kurtosis(freq_series, bias=False)
            sk_values.append(sk)
        else:
            sk_values.append(0)
    
    sk_values = np.array(sk_values)
    
    # 提取原有谱峭度特征
    features.append(np.max(sk_values))  # 最大谱峭度
    max_sk_idx = np.argmax(sk_values)
    features.append(f[max_sk_idx])  # 最大谱峭度对应的频率
    
    # 频带谱峭度均值
    n_bands = 5
    band_size = len(sk_values) // n_bands
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < n_bands - 1 else len(sk_values)
        band_sk = np.mean(sk_values[start_idx:end_idx])
        features.append(band_sk)
    
    # 新增：Kurtogram最佳频带选择特征
    try:
        best_signal, best_kurt, freq_range = lightweight_kurtogram_band_selection(signal_data, fs)
        
        # 最佳频带特征
        features.append(best_kurt)  # 最佳频带峭度
        features.append(freq_range[0])  # 最佳频带下限频率
        features.append(freq_range[1])  # 最佳频带上限频率
        features.append(freq_range[1] - freq_range[0])  # 最佳频带宽度
        
        # 最佳频带信号的时域特征
        if len(best_signal) > 3 and np.std(best_signal) > 1e-10:
            features.append(np.std(best_signal))  # 最佳频带标准差
            features.append(np.sqrt(np.mean(best_signal**2)))  # 最佳频带RMS
            features.append(np.max(np.abs(best_signal)))  # 最佳频带峰值
        else:
            features.extend([0, 0, 0])
        
    except Exception as e:
        print(f"Kurtogram特征提取出错: {e}")
        # 如果出错，添加默认值
        features.extend([0, 0, 0, 0, 0, 0, 0])
    
    return features

# 倒谱特征提取
def extract_cepstrum_features(signal_data, fs=SAMPLE_RATE):
    """提取倒谱特征"""
    features = []
    
    # 计算功率谱
    fft_result = np.abs(np.fft.fft(signal_data))
    fft_result = fft_result[:len(fft_result)//2]
    
    # 避免log(0)
    fft_result = np.maximum(fft_result, 1e-10)
    
    # 计算倒谱
    log_spectrum = np.log(fft_result)
    cepstrum = np.abs(np.fft.ifft(log_spectrum))
    cepstrum = cepstrum[:len(cepstrum)//2]
    
    # 低quefrency峰值（转频附近）
    low_q_end = min(50, len(cepstrum)//4)
    low_q_peak = np.max(cepstrum[:low_q_end])
    features.append(low_q_peak)
    
    # 中quefrency峰值（滚动体故障调制）
    mid_q_start = low_q_end
    mid_q_end = min(len(cepstrum)//2, mid_q_start + 100)
    if mid_q_end > mid_q_start:
        mid_q_peak = np.max(cepstrum[mid_q_start:mid_q_end])
        features.append(mid_q_peak)
    else:
        features.append(0)
    
    # 能量比
    total_energy = np.sum(cepstrum**2)
    if total_energy > 0:
        low_energy_ratio = np.sum(cepstrum[:low_q_end]**2) / total_energy
        features.append(low_energy_ratio)
    else:
        features.append(0)
    
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

# 频域特征提取
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

# 包络分析特征（增强版）
def extract_envelope_features(signal_data, fs=SAMPLE_RATE):
    """提取包络分析特征（增强版）"""
    features = []
    
    # 计算希尔伯特变换得到解析信号
    analytic_signal = signal.hilbert(signal_data)
    
    # 计算包络
    envelope = np.abs(analytic_signal)
    
    # 包络的基本统计特征
    features.append(np.mean(envelope))  # 包络均值
    features.append(np.std(envelope))   # 包络标准差
    features.append(np.max(envelope))   # 包络峰值
    
    # 包络的均方根值
    env_rms = np.sqrt(np.mean(np.square(envelope)))
    features.append(env_rms)  # 包络RMS
    
    # 新增：包络的高阶统计特征
    features.append(skew(envelope))     # 包络偏度
    features.append(kurtosis(envelope)) # 包络峭度
    
    # 新增：包络的形状因子
    env_mean_abs = np.mean(np.abs(envelope))
    env_peak = np.max(envelope)
    if env_mean_abs > 0:
        features.append(env_peak / env_mean_abs)  # 包络脉冲因子
    else:
        features.append(0)
    
    if env_rms > 0:
        features.append(env_rms / env_mean_abs if env_mean_abs > 0 else 0)  # 包络波形因子
        features.append(env_peak / env_rms)  # 包络峭度因子
    else:
        features.extend([0, 0])
    
    # 包络的频谱分析
    env_fft = np.abs(np.fft.fft(envelope))
    env_fft = env_fft[:len(env_fft)//2]  # 取前半部分
    
    # 包络频谱的主频及其幅值
    freq_resolution = fs / len(envelope)
    frequencies = np.arange(len(env_fft)) * freq_resolution
    
    dominant_idx = np.argmax(env_fft)
    dominant_freq = frequencies[dominant_idx]
    dominant_amp = env_fft[dominant_idx]
    
    features.append(dominant_freq)  # 包络主频
    features.append(dominant_amp)   # 包络主频幅值
    
    # 包络频谱能量
    total_env_energy = np.sum(env_fft**2)
    features.append(total_env_energy)  # 包络频谱总能量
    
    # 新增：包络频谱的分布特征
    # 包络频谱质心
    if np.sum(env_fft) > 0:
        env_spectral_centroid = np.sum(frequencies * env_fft) / np.sum(env_fft)
        features.append(env_spectral_centroid)
    else:
        features.append(0)
    
    # 包络频谱熵
    normalized_env_fft = env_fft / np.sum(env_fft) if np.sum(env_fft) > 0 else np.zeros_like(env_fft)
    env_spectral_entropy = -np.sum(normalized_env_fft * np.log2(normalized_env_fft + 1e-10))
    features.append(env_spectral_entropy)
    
    # 新增：包络频谱的频带能量分布
    # 将包络频谱分为3个频带：低频、中频、高频
    n_bands = 3
    band_size = len(env_fft) // n_bands
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < n_bands - 1 else len(env_fft)
        band_energy = np.sum(env_fft[start_idx:end_idx]**2)
        features.append(band_energy / total_env_energy if total_env_energy > 0 else 0)
    
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
            
            # 提取谱峭度特征
            sk_features = extract_spectral_kurtosis_features(window)
            window_features.extend(sk_features)
            
            # 提取倒谱特征
            cepstrum_features = extract_cepstrum_features(window)
            window_features.extend(cepstrum_features)
            
            # 提取包络谱侧带特征
            sideband_features = extract_envelope_sideband_features(window)
            window_features.extend(sideband_features)
            
            all_features.append(window_features)
        
        # 多统计量聚合替代简单平均
        F = np.vstack(all_features)  # (n_win, n_feat)
        agg_features = np.concatenate([
            F.mean(0),                           # 均值
            F.std(0),                            # 标准差
            F.max(0),                            # 最大值
            np.quantile(F, 0.95, axis=0),        # 95分位数
            skew(F, axis=0, bias=False),         # 偏度
            kurtosis(F, axis=0, bias=False)      # 峰度
        ])
        
        return agg_features
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
    
    # 确保X是2D数组
    X_array = np.array(X)
    if len(X_array.shape) == 1:
        # 如果是1D数组，将其转换为2D数组
        X_array = X_array.reshape(-1, 1)
    
    print(f"特征数组形状: {X_array.shape}")
    return X_array, np.array(y)

# 特征选择和降维
def feature_selection(X, y, variance_threshold=0.01, correlation_threshold=0.95, n_components=None):
    """特征选择和降维（添加相关性去冗余）"""
    # 方差过滤
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var_filtered = selector.fit_transform(X)
    
    print(f"方差过滤后特征数: {X_var_filtered.shape[1]}")
    
    # 相关性去冗余
    if X_var_filtered.shape[1] > 1:
        # 计算特征间相关性矩阵
        corr_matrix = np.corrcoef(X_var_filtered.T)
        
        # 找到高相关性的特征对
        high_corr_pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    high_corr_pairs.append((i, j, abs(corr_matrix[i, j])))
        
        # 移除冗余特征（保留第一个，移除后续相关的）
        features_to_remove = set()
        for i, j, corr_val in high_corr_pairs:
            if i not in features_to_remove:
                features_to_remove.add(j)
        
        # 创建保留特征的索引
        features_to_keep = [i for i in range(X_var_filtered.shape[1]) if i not in features_to_remove]
        X_corr_filtered = X_var_filtered[:, features_to_keep]
        
        print(f"相关性去冗余后特征数: {X_corr_filtered.shape[1]} (移除了{len(features_to_remove)}个高相关特征)")
    else:
        X_corr_filtered = X_var_filtered
        features_to_keep = list(range(X_var_filtered.shape[1]))
    
    # 如果指定了PCA组件数，则进行PCA降维
    if n_components is not None and n_components < X_corr_filtered.shape[1]:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_corr_filtered)
        print(f"PCA降维后特征数: {X_reduced.shape[1]}")
        print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")
        return X_reduced, (selector, features_to_keep), pca

    return X_corr_filtered, (selector, features_to_keep), None


def apply_feature_selection_and_pca(X, selector=None, pca=None):
    """将特征选择器和PCA变换统一应用于数据。"""
    if selector is not None:
        if isinstance(selector, tuple):
            variance_selector, features_to_keep = selector
            X = variance_selector.transform(X)
            X = X[:, features_to_keep]
        else:
            X = selector.transform(X)

    if pca is not None:
        X = pca.transform(X)

    return X


def create_rf_model(best_params=None, random_state=42):
    """根据给定参数构建随机森林模型。"""
    base_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'oob_score': True,
        'class_weight': 'balanced',
        'random_state': random_state,
        'n_jobs': -1,
    }
    if best_params:
        base_params.update(best_params)
        # 确保关键信息仍然存在
        base_params.setdefault('class_weight', 'balanced')
        base_params.setdefault('random_state', random_state)
        base_params.setdefault('n_jobs', -1)

    return RandomForestClassifier(**base_params)


def create_xgb_model(best_params=None, random_state=42):
    """根据给定参数构建XGBoost模型。"""
    base_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.1,
        'min_child_weight': 3,
        'objective': 'multi:softprob',
        'num_class': len(fault_types),
        'random_state': random_state,
        'eval_metric': 'mlogloss',
        'scale_pos_weight': None,
    }

    if best_params:
        base_params.update(best_params)
        base_params.setdefault('objective', 'multi:softprob')
        base_params.setdefault('num_class', len(fault_types))
        base_params.setdefault('eval_metric', 'mlogloss')

    return xgb.XGBClassifier(**base_params)


def cross_validate_model(X, y, build_model_fn, model_label="model", n_splits=5,
                         variance_threshold=0.01, correlation_threshold=0.95):
    """对给定模型进行分层交叉验证评估以估计可靠性。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    macro_f1_scores = []
    log_losses = []

    print(f"\n对模型 {model_label} 进行{n_splits}折交叉验证评估...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_selected, selector_info, pca = feature_selection(
            X_train_scaled, y_train,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            n_components=None
        )
        X_val_selected = apply_feature_selection_and_pca(X_val_scaled, selector_info, pca)

        model = build_model_fn()

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        try:
            model.fit(X_train_selected, y_train, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_train_selected, y_train)

        y_pred = model.predict(X_val_selected)
        y_proba = model.predict_proba(X_val_selected)

        y_proba = np.clip(y_proba, 1e-8, 1 - 1e-8)
        y_proba = y_proba / np.sum(y_proba, axis=1, keepdims=True)

        fold_acc = accuracy_score(y_val, y_pred)
        fold_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        fold_log_loss = log_loss(y_val, y_proba, labels=list(range(len(fault_types))))

        accuracies.append(fold_acc)
        macro_f1_scores.append(fold_f1)
        log_losses.append(fold_log_loss)

        print(f"  折 {fold_idx}: 准确率={fold_acc:.4f}, Macro-F1={fold_f1:.4f}, LogLoss={fold_log_loss:.4f}")

    metrics = {
        'accuracy_mean': float(np.mean(accuracies)) if accuracies else 0.0,
        'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
        'macro_f1_mean': float(np.mean(macro_f1_scores)) if macro_f1_scores else 0.0,
        'macro_f1_std': float(np.std(macro_f1_scores)) if macro_f1_scores else 0.0,
        'log_loss_mean': float(np.mean(log_losses)) if log_losses else None,
        'log_loss_std': float(np.std(log_losses)) if log_losses else None,
        'n_splits': n_splits,
        'model_label': model_label,
    }

    if metrics['log_loss_mean'] is not None:
        log_loss_text = f"{metrics['log_loss_mean']:.4f}±{metrics['log_loss_std']:.4f}"
    else:
        log_loss_text = "nan"

    print(
        f"{model_label} 交叉验证: 平均准确率={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f}, "
        f"平均Macro-F1={metrics['macro_f1_mean']:.4f}±{metrics['macro_f1_std']:.4f}, 平均LogLoss={log_loss_text}"
    )

    return metrics


def compute_reliability_weight(metrics):
    """根据交叉验证指标计算模型全局可靠性权重。"""
    accuracy = metrics.get('accuracy_mean', 0.0) or 0.0
    log_loss_value = metrics.get('log_loss_mean')

    weight = max(accuracy, 1e-6)
    if log_loss_value is not None:
        weight *= np.exp(-max(log_loss_value, 0.0))

    return float(weight)

# 超参数调优函数
def hyperparameter_tuning(X, y, model_type='rf', use_grid_search=True):
    """使用网格搜索或随机搜索进行超参数调优
    
    参数:
        X: 特征数据
        y: 标签数据
        model_type: 模型类型 ('rf' 或 'xgb')
        use_grid_search: 是否使用网格搜索（否则使用随机搜索）
    
    返回:
        最佳参数字典
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择
    X_selected, _, _ = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    if model_type == 'rf':
        # 随机森林超参数网格
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            search_cv = GridSearchCV(
                RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
                param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2', None]
            }
            search_cv = RandomizedSearchCV(
                RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
                param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
    
    elif model_type == 'xgb':
        # 计算样本权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            search_cv = GridSearchCV(
                xgb.XGBClassifier(objective='multi:softprob', num_class=len(fault_types), random_state=42),
                param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 250],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0.1, 1, 10]
            }
            search_cv = RandomizedSearchCV(
                xgb.XGBClassifier(objective='multi:softprob', num_class=len(fault_types), random_state=42),
                param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
            )
        
        # XGBoost需要样本权重
        search_cv.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        raise ValueError("model_type必须是'rf'或'xgb'")
    
    if model_type == 'rf':
        search_cv.fit(X_train, y_train)
    
    print(f"\n{model_type.upper()}最佳参数: {search_cv.best_params_}")
    print(f"{model_type.upper()}最佳交叉验证得分: {search_cv.best_score_:.4f}")
    
    return search_cv.best_params_

# 训练随机森林模型（增强版）
def train_rf_model_enhanced(X, y, best_params=None):
    """训练随机森林分类器（增强版，支持超参数调优）"""
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集（添加分层采样）
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # 计算类别权重处理不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"随机森林类别权重: {class_weight_dict}")

    # 使用最佳参数或默认参数
    if best_params is not None:
        print(f"使用调优后的参数: {best_params}")

    model = create_rf_model(best_params)

    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_train, y_train)

    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"随机森林验证集准确率: {accuracy:.4f}")
    if hasattr(model, 'oob_score_'):
        print(f"随机森林袋外得分: {model.oob_score_:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("随机森林特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca

# 训练XGBoost模型（增强版）
def train_xgb_model_enhanced(X, y, best_params=None):
    """训练XGBoost分类器（增强版，支持超参数调优）"""
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集（添加分层采样）
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # 计算类别权重处理不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # 计算样本权重
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    print(f"XGBoost类别权重: {class_weight_dict}")

    # 使用最佳参数或默认参数
    if best_params is not None:
        print(f"使用调优后的参数: {best_params}")

    model = create_xgb_model(best_params)

    # 使用样本权重训练
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"XGBoost验证集准确率: {accuracy:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("XGBoost特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca

# 训练随机森林模型
def train_rf_model(X, y):
    """训练随机森林分类器（添加类不平衡处理）"""
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集（添加分层采样）
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # 计算类别权重处理不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"随机森林类别权重: {class_weight_dict}")

    # 训练随机森林模型（优化参数 + 类不平衡处理）
    model = create_rf_model()

    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_train, y_train)

    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"随机森林验证集准确率: {accuracy:.4f}")
    if hasattr(model, 'oob_score_'):
        print(f"随机森林袋外得分: {model.oob_score_:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("随机森林特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca

# 训练XGBoost模型
def train_xgb_model(X, y):
    """训练XGBoost分类器（添加类不平衡处理）"""
    # 数据标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集（添加分层采样）
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    # 计算类别权重处理不平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # 计算样本权重
    sample_weights = np.array([class_weight_dict[label] for label in y_train])
    
    print(f"XGBoost类别权重: {class_weight_dict}")

    # 训练XGBoost模型（优化参数 + 类不平衡处理）
    model = create_xgb_model()

    # 使用样本权重训练
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"XGBoost验证集准确率: {accuracy:.4f}")
    print(classification_report(y_val, y_pred))
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("XGBoost特征重要性 (前10):")
        for i in range(min(10, len(importances))):
            print(f"特征 #{indices[i]}: {importances[indices[i]]:.4f}")
    
    return model, scaler, selector, pca

# 预测测试数据
def predict_test_data_with_proba(model, scaler, selector=None, pca=None, use_order_tracking=True, use_windowing=True, model_name="RandomForest"):
    """预测测试数据集中的故障类型（返回概率）
    
    参数:
        model: 训练好的模型
        scaler: 标准化器
        selector: 特征选择器 (可选)
        pca: PCA降维器 (可选)
        use_order_tracking: 是否使用阶次处理
        use_windowing: 是否使用窗口处理
        model_name: 模型名称
    
    返回:
        results: 预测结果列表 [(文件名, 故障类型)]
        probabilities: 概率字典 {文件名: 概率数组}
    """
    test_files = glob.glob(os.path.join(test_path, "*.xlsx"))
    results = []
    probabilities = {}
    
    print(f"使用{model_name}模型预测测试数据...")
    
    for file in test_files:
        file_name = os.path.basename(file)
        features = extract_features(file, use_order_tracking, use_windowing)
        
        if features is not None:
            # 标准化特征
            features_scaled = scaler.transform([features])

            features_scaled = apply_feature_selection_and_pca(features_scaled, selector, pca)
                
            # 预测故障类型和概率
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            # 获取故障类型名称
            fault_name = list(fault_types.keys())[list(fault_types.values()).index(prediction)]
            # 去掉后缀，只保留故障类型
            fault_name = fault_name.split('_train')[0]
            
            results.append((file_name, fault_name))
            probabilities[file_name] = proba
            print(f"文件: {file_name}, 预测故障类型: {fault_name}")
    
    # 按文件名排序
    results.sort(key=lambda x: x[0])
    
    return results, probabilities

def predict_test_data(model, scaler, selector=None, pca=None, use_order_tracking=True, use_windowing=True, model_name="RandomForest"):
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
    
    print(f"使用{model_name}模型预测测试数据...")
    
    for file in test_files:
        file_name = os.path.basename(file)
        features = extract_features(file, use_order_tracking, use_windowing)
        
        if features is not None:
            # 标准化特征
            features_scaled = scaler.transform([features])

            features_scaled = apply_feature_selection_and_pca(features_scaled, selector, pca)
                
            # 预测故障类型
            prediction = model.predict(features_scaled)[0]
            # 获取故障类型名称
            fault_name = list(fault_types.keys())[list(fault_types.values()).index(prediction)]
            # 去掉后缀，只保留故障类型
            fault_name = fault_name.split('_train')[0]
            
            results.append((file_name, fault_name))
            print(f"文件: {file_name}, 预测故障类型: {fault_name}")
    
    # 按文件名排序
    results.sort(key=lambda x: x[0])
    
    return results

# 保存预测结果
def save_results(results, output_file="prediction_results.txt"):
    """保存预测结果到文件"""
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    with open(output_file, 'w') as f:
        for file_name, fault_type in results:
            f.write(f"{file_name},{fault_type}\n")
    print(f"预测结果已保存到 {output_file}")

# 集成多个模型的预测结果
def ensemble_predictions_with_proba(rf_probabilities, xgb_probabilities, output_file="prediction_results.txt", reliability_weights=None):
    """基于概率的集成预测（优化版）

    参数:
        rf_probabilities: 随机森林概率字典 {文件名: 概率数组}
        xgb_probabilities: XGBoost概率字典 {文件名: 概率数组}
        output_file: 输出文件名
        reliability_weights: 全局模型可靠性权重字典，如 {'rf': 1.0, 'xgb': 0.9}

    返回:
        集成后的预测结果列表
    """
    # 获取所有文件名
    all_files = sorted(list(set(list(rf_probabilities.keys()) + list(xgb_probabilities.keys()))))
    
    # 集成结果
    ensemble_results = []
    
    # 动态权重策略：基于置信度的自适应权重
    for file in all_files:
        rf_proba = rf_probabilities.get(file, None)
        xgb_proba = xgb_probabilities.get(file, None)
        
        if rf_proba is not None and xgb_proba is not None:
            # 计算每个模型的置信度（最大概率值）
            rf_confidence = np.max(rf_proba)
            xgb_confidence = np.max(xgb_proba)
            
            # 计算熵作为不确定性度量
            rf_entropy = -np.sum(rf_proba * np.log(rf_proba + 1e-10))
            xgb_entropy = -np.sum(xgb_proba * np.log(xgb_proba + 1e-10))
            
            # 基于置信度和熵的动态权重
            # 高置信度、低熵的模型获得更高权重
            rf_weight = rf_confidence / (rf_entropy + 1e-10)
            xgb_weight = xgb_confidence / (xgb_entropy + 1e-10)

            if reliability_weights:
                rf_weight *= reliability_weights.get('rf', 1.0)
                xgb_weight *= reliability_weights.get('xgb', 1.0)

            # 归一化权重
            total_weight = rf_weight + xgb_weight
            rf_weight_norm = rf_weight / total_weight
            xgb_weight_norm = xgb_weight / total_weight
            
            # 加权融合
            avg_proba = rf_weight_norm * rf_proba + xgb_weight_norm * xgb_proba
            
            # 温度缩放（Temperature Scaling）提高预测置信度
            temperature = 0.8  # 小于1使分布更尖锐
            avg_proba = np.exp(np.log(avg_proba + 1e-10) / temperature)
            avg_proba = avg_proba / np.sum(avg_proba)  # 重新归一化
            
        elif xgb_proba is not None:
            avg_proba = xgb_proba
        elif rf_proba is not None:
            avg_proba = rf_proba
        else:
            continue
            
        # 获取最高概率对应的类别
        prediction = np.argmax(avg_proba)
        
        # 获取故障类型名称
        fault_name = list(fault_types.keys())[list(fault_types.values()).index(prediction)]
        # 去掉后缀，只保留故障类型
        fault_name = fault_name.split('_train')[0]
        
        ensemble_results.append((file, fault_name))
        
        # 显示详细融合信息
        if rf_proba is not None and xgb_proba is not None:
            print(f"文件: {file}, RF权重: {rf_weight_norm:.3f}, XGB权重: {xgb_weight_norm:.3f}, "
                  f"融合预测: {fault_name} (置信度: {avg_proba[prediction]:.3f})")
        else:
            print(f"文件: {file}, 单模型预测: {fault_name} (置信度: {avg_proba[prediction]:.3f})")
    
    # 保存结果
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    
    with open(output_file, 'w') as f:
        for file_name, fault_type in ensemble_results:
            f.write(f"{file_name},{fault_type}\n")
    
    print(f"优化概率融合预测结果已保存到 {output_file}")
    return ensemble_results

# 主函数（增强版）
def main_enhanced():
    """主函数，执行完整的故障预测流程（增强版）"""
    print("开始故障预测任务（增强版）...")
    
    # 设置参数
    use_order_tracking = True  # 是否使用阶次处理
    use_windowing = False      # 去掉窗口处理，专注于其他优化
    use_hyperparameter_tuning = True  # 是否进行超参数调优
    
    print(f"参数设置: 阶次处理={use_order_tracking}, 窗口处理={use_windowing}, 超参数调优={use_hyperparameter_tuning}")
    
    # 加载训练数据
    print("正在加载训练数据...")
    X, y = load_training_data(use_order_tracking, use_windowing)
    
    # 超参数调优（可选）
    rf_best_params = None
    xgb_best_params = None

    if use_hyperparameter_tuning:
        print("\n进行随机森林超参数调优...")
        try:
            rf_best_params = hyperparameter_tuning(X, y, model_type='rf', use_grid_search=False)
        except Exception as e:
            print(f"随机森林超参数调优失败: {e}")
            rf_best_params = None
        
        print("\n进行XGBoost超参数调优...")
        try:
            xgb_best_params = hyperparameter_tuning(X, y, model_type='xgb', use_grid_search=False)
        except Exception as e:
            print(f"XGBoost超参数调优失败: {e}")
            xgb_best_params = None

    reliability_weights = {'rf': 1.0, 'xgb': 1.0}
    print("\n执行交叉验证以评估模型可靠性...")
    try:
        rf_cv_metrics = cross_validate_model(
            X, y,
            build_model_fn=lambda: create_rf_model(rf_best_params),
            model_label="RandomForest-Enhanced"
        )
        reliability_weights['rf'] = compute_reliability_weight(rf_cv_metrics)
    except Exception as e:
        print(f"随机森林交叉验证失败，使用默认权重: {e}")

    try:
        xgb_cv_metrics = cross_validate_model(
            X, y,
            build_model_fn=lambda: create_xgb_model(xgb_best_params),
            model_label="XGBoost-Enhanced"
        )
        reliability_weights['xgb'] = compute_reliability_weight(xgb_cv_metrics)
    except Exception as e:
        print(f"XGBoost交叉验证失败，使用默认权重: {e}")

    print(f"模型可靠性权重: {reliability_weights}")

    # 训练增强版随机森林模型
    print("\n训练增强版随机森林模型...")
    rf_model, rf_scaler, rf_selector, rf_pca = train_rf_model_enhanced(X, y, rf_best_params)

    # 训练增强版XGBoost模型
    print("\n训练增强版XGBoost模型...")
    xgb_model, xgb_scaler, xgb_selector, xgb_pca = train_xgb_model_enhanced(X, y, xgb_best_params)
    
    # 使用随机森林模型预测测试数据（获取概率）
    print("\n使用增强版随机森林模型预测测试数据...")
    rf_results, rf_probabilities = predict_test_data_with_proba(rf_model, rf_scaler, rf_selector, rf_pca, 
                                                               use_order_tracking, use_windowing, "RandomForest-Enhanced")
    
    # 保存随机森林模型的预测结果
    save_results(rf_results, "rf_enhanced_prediction_results.txt")
    
    # 使用XGBoost模型预测测试数据（获取概率）
    print("\n使用增强版XGBoost模型预测测试数据...")
    xgb_results, xgb_probabilities = predict_test_data_with_proba(xgb_model, xgb_scaler, xgb_selector, xgb_pca, 
                                                                 use_order_tracking, use_windowing, "XGBoost-Enhanced")
    
    # 保存XGBoost模型的预测结果
    save_results(xgb_results, "xgb_enhanced_prediction_results.txt")
    
    # 集成两个模型的预测结果（使用增强的概率融合）
    print("\n集成两个模型的预测结果（增强概率融合）...")
    ensemble_results = ensemble_predictions_with_proba(
        rf_probabilities,
        xgb_probabilities,
        "enhanced_prediction_results.txt",
        reliability_weights=reliability_weights
    )
    
    # 保存按照提交格式的结果
    save_results_for_submission(ensemble_results, "enhanced_submit_result.txt")
    
    print("\n增强版故障预测任务完成!")

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
    
    # 训练随机森林模型
    print("\n训练随机森林模型...")
    rf_model, rf_scaler, rf_selector, rf_pca = train_rf_model(X, y)
    
    # 训练XGBoost模型
    print("\n训练XGBoost模型...")
    xgb_model, xgb_scaler, xgb_selector, xgb_pca = train_xgb_model(X, y)
    
    # 使用随机森林模型预测测试数据（获取概率）
    print("\n使用随机森林模型预测测试数据...")
    rf_results, rf_probabilities = predict_test_data_with_proba(rf_model, rf_scaler, rf_selector, rf_pca, 
                                                               use_order_tracking, use_windowing, "RandomForest")
    
    # 保存随机森林模型的预测结果
    save_results(rf_results, "rf_prediction_results.txt")
    
    # 使用XGBoost模型预测测试数据（获取概率）
    print("\n使用XGBoost模型预测测试数据...")
    xgb_results, xgb_probabilities = predict_test_data_with_proba(xgb_model, xgb_scaler, xgb_selector, xgb_pca, 
                                                                 use_order_tracking, use_windowing, "XGBoost")
    
    # 保存XGBoost模型的预测结果
    save_results(xgb_results, "xgb_prediction_results.txt")
    
    # 集成两个模型的预测结果（使用概率融合）
    print("\n集成两个模型的预测结果（概率融合）...")
    ensemble_results = ensemble_predictions_with_proba(rf_probabilities, xgb_probabilities, "prediction_results.txt")
    
    # 保存按照提交格式的结果
    save_results_for_submission(ensemble_results)
    
    print("\n故障预测任务完成!")


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


# 程序入口
if __name__ == "__main__":
    # 运行增强版主函数
    main_enhanced()