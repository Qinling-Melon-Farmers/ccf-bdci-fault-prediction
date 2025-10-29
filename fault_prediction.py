import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import glob
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# 定义数据路径（使用脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(PROJECT_ROOT, "初赛数据集(6种)")
train_path = os.path.join(base_path, "初赛训练集")
test_path = os.path.join(base_path, "初赛测试集")

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

# 包络分析特征
def extract_envelope_features(signal_data, fs=SAMPLE_RATE):
    """提取包络分析特征"""
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
    features.append(np.sum(env_fft**2))  # 包络频谱总能量
    
    return features

# 小波特征提取
def extract_wavelet_features(signal_data):
    """提取小波特征"""
    features = []
    
    # 使用小波变换的多尺度分解
    # 这里使用简化版本，计算不同尺度下的能量分布
    
    # 定义几个尺度级别
    scales = [2, 4, 8, 16, 32]
    
    for scale in scales:
        # 使用简单的均值滤波模拟小波分解的不同尺度
        kernel = np.ones(scale) / scale
        # 对信号进行卷积
        filtered = np.convolve(signal_data, kernel, mode='same')
        # 计算该尺度下的能量
        energy = np.sum(filtered**2)
        features.append(energy)
    
    # 计算各尺度能量的分布
    total_energy = np.sum(features)
    if total_energy > 0:
        normalized_energies = [e / total_energy for e in features]
        features.extend(normalized_energies)  # 添加归一化的能量分布
    else:
        features.extend([0] * len(scales))
    
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
    
    # 确保X是2D数组
    X_array = np.array(X)
    if len(X_array.shape) == 1:
        # 如果是1D数组，将其转换为2D数组
        X_array = X_array.reshape(-1, 1)
    
    print(f"特征数组形状: {X_array.shape}")
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
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"随机森林验证集准确率: {accuracy:.4f}")
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
    """训练XGBoost分类器"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择和降维
    X_selected, selector, pca = feature_selection(X_scaled, y, variance_threshold=0.01, n_components=None)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
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
def ensemble_predictions(rf_results, xgb_results, output_file="prediction_results.txt"):
    """集成随机森林和XGBoost模型的预测结果
    
    参数:
        rf_results: 随机森林模型的预测结果列表，每项为(文件名, 预测类型)
        xgb_results: XGBoost模型的预测结果列表，每项为(文件名, 预测类型)
        output_file: 输出文件名
    
    返回:
        集成后的预测结果列表
    """
    # 创建文件名到预测结果的映射
    rf_dict = {file: fault for file, fault in rf_results}
    xgb_dict = {file: fault for file, fault in xgb_results}
    # 获取所有文件名
    all_files = sorted(list(set(list(rf_dict.keys()) + list(xgb_dict.keys()))))
    # 集成结果
    ensemble_results = []
    for file in all_files:
        rf_pred = rf_dict.get(file, None)
        xgb_pred = xgb_dict.get(file, None)
        if rf_pred and xgb_pred and rf_pred != xgb_pred:
            final_pred = xgb_pred
            print(f"文件: {file}, RF预测: {rf_pred}, XGB预测: {xgb_pred}, 最终选择: {final_pred}")
        else:
            final_pred = xgb_pred if xgb_pred else rf_pred
        ensemble_results.append((file, final_pred))
    if not os.path.isabs(output_file):
        output_file = os.path.join(PROJECT_ROOT, output_file)
    with open(output_file, 'w') as f:
        for file_name, fault_type in ensemble_results:
            f.write(f"{file_name},{fault_type}\n")
    print(f"集成预测结果已保存到 {output_file}")
    return ensemble_results

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
    
    # 使用随机森林模型预测测试数据
    print("\n使用随机森林模型预测测试数据...")
    rf_results = predict_test_data(rf_model, rf_scaler, rf_selector, rf_pca, 
                                  use_order_tracking, use_windowing, "RandomForest")
    
    # 保存随机森林模型的预测结果
    save_results(rf_results, "rf_prediction_results.txt")
    
    # 使用XGBoost模型预测测试数据
    print("\n使用XGBoost模型预测测试数据...")
    xgb_results = predict_test_data(xgb_model, xgb_scaler, xgb_selector, xgb_pca, 
                                   use_order_tracking, use_windowing, "XGBoost")
    
    # 保存XGBoost模型的预测结果
    save_results(xgb_results, "xgb_prediction_results.txt")
    
    # 集成两个模型的预测结果
    print("\n集成两个模型的预测结果...")
    ensemble_results = ensemble_predictions(rf_results, xgb_results, "prediction_results.txt")
    
    # 保存按照提交格式的结果
    save_results_for_submission(ensemble_results)
    
    print("\n故障预测任务完成!")

# 程序入口
if __name__ == "__main__":
    main()


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