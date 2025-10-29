"""
特征工程模块
添加统计特征、频域特征、时域特征等
"""
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, sampling_rate=1000):
        """
        初始化特征提取器
        
        Args:
            sampling_rate: 采样率
        """
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def extract_statistical_features(self, data):
        """
        提取统计特征
        
        Args:
            data: 输入数据 (batch_size, seq_len, channels)
            
        Returns:
            numpy.ndarray: 统计特征
        """
        features = []
        
        for i in range(data.shape[0]):  # 遍历每个样本
            sample_features = []
            
            for j in range(data.shape[2]):  # 遍历每个通道
                channel_data = data[i, :, j]
                
                # 基本统计特征
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                var_val = np.var(channel_data)
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                range_val = max_val - min_val
                
                # 分位数特征
                q25 = np.percentile(channel_data, 25)
                q50 = np.percentile(channel_data, 50)  # 中位数
                q75 = np.percentile(channel_data, 75)
                iqr = q75 - q25
                
                # 高阶统计特征
                skewness = stats.skew(channel_data)
                kurtosis = stats.kurtosis(channel_data)
                
                # 能量特征
                energy = np.sum(channel_data ** 2)
                rms = np.sqrt(np.mean(channel_data ** 2))
                
                # 零交叉率
                zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                
                # 峰值特征
                peaks, _ = signal.find_peaks(channel_data)
                num_peaks = len(peaks)
                
                # 梯度特征
                gradient = np.gradient(channel_data)
                mean_gradient = np.mean(np.abs(gradient))
                std_gradient = np.std(gradient)
                
                sample_features.extend([
                    mean_val, std_val, var_val, min_val, max_val, range_val,
                    q25, q50, q75, iqr, skewness, kurtosis,
                    energy, rms, zero_crossings, num_peaks,
                    mean_gradient, std_gradient
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_frequency_features(self, data):
        """
        提取频域特征
        
        Args:
            data: 输入数据 (batch_size, seq_len, channels)
            
        Returns:
            numpy.ndarray: 频域特征
        """
        features = []
        
        for i in range(data.shape[0]):  # 遍历每个样本
            sample_features = []
            
            for j in range(data.shape[2]):  # 遍历每个通道
                channel_data = data[i, :, j]
                
                # FFT变换
                fft_vals = fft(channel_data)
                fft_magnitude = np.abs(fft_vals)
                fft_phase = np.angle(fft_vals)
                freqs = fftfreq(len(channel_data), 1/self.sampling_rate)
                
                # 只取正频率部分
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                
                # 频域统计特征
                spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
                spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_magnitude) / np.sum(positive_magnitude))
                spectral_rolloff = self._spectral_rolloff(positive_freqs, positive_magnitude)
                spectral_flux = np.sum(np.diff(positive_magnitude) ** 2)
                
                # 功率谱密度
                psd = positive_magnitude ** 2
                total_power = np.sum(psd)
                
                # 频带能量比
                low_freq_power = np.sum(psd[positive_freqs < 50])  # 低频能量
                mid_freq_power = np.sum(psd[(positive_freqs >= 50) & (positive_freqs < 200)])  # 中频能量
                high_freq_power = np.sum(psd[positive_freqs >= 200])  # 高频能量
                
                low_freq_ratio = low_freq_power / total_power if total_power > 0 else 0
                mid_freq_ratio = mid_freq_power / total_power if total_power > 0 else 0
                high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
                
                # 主频率
                dominant_freq = positive_freqs[np.argmax(positive_magnitude)]
                
                # 频谱熵
                normalized_psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
                spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
                
                sample_features.extend([
                    spectral_centroid, spectral_spread, spectral_rolloff, spectral_flux,
                    total_power, low_freq_ratio, mid_freq_ratio, high_freq_ratio,
                    dominant_freq, spectral_entropy
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def _spectral_rolloff(self, freqs, magnitude, rolloff_percent=0.85):
        """计算频谱滚降点"""
        total_energy = np.sum(magnitude)
        cumulative_energy = np.cumsum(magnitude)
        rolloff_index = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
        
        if len(rolloff_index) > 0:
            return freqs[rolloff_index[0]]
        else:
            return freqs[-1]
    
    def extract_time_domain_features(self, data):
        """
        提取时域特征
        
        Args:
            data: 输入数据 (batch_size, seq_len, channels)
            
        Returns:
            numpy.ndarray: 时域特征
        """
        features = []
        
        for i in range(data.shape[0]):  # 遍历每个样本
            sample_features = []
            
            for j in range(data.shape[2]):  # 遍历每个通道
                channel_data = data[i, :, j]
                
                # 自相关特征
                autocorr = np.correlate(channel_data, channel_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                
                # 自相关峰值
                autocorr_peaks, _ = signal.find_peaks(autocorr[1:])  # 排除第一个点
                num_autocorr_peaks = len(autocorr_peaks)
                
                # 波形因子
                form_factor = np.sqrt(np.mean(channel_data ** 2)) / np.mean(np.abs(channel_data)) if np.mean(np.abs(channel_data)) != 0 else 0
                
                # 峰值因子
                crest_factor = np.max(np.abs(channel_data)) / np.sqrt(np.mean(channel_data ** 2)) if np.sqrt(np.mean(channel_data ** 2)) != 0 else 0
                
                # 脉冲因子
                impulse_factor = np.max(np.abs(channel_data)) / np.mean(np.abs(channel_data)) if np.mean(np.abs(channel_data)) != 0 else 0
                
                # 裕度因子
                clearance_factor = np.max(np.abs(channel_data)) / (np.mean(np.sqrt(np.abs(channel_data))) ** 2) if np.mean(np.sqrt(np.abs(channel_data))) != 0 else 0
                
                # 形状因子
                shape_factor = np.sqrt(np.mean(channel_data ** 2)) / np.mean(np.abs(channel_data)) if np.mean(np.abs(channel_data)) != 0 else 0
                
                # 趋势特征
                x = np.arange(len(channel_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, channel_data)
                
                sample_features.extend([
                    num_autocorr_peaks, form_factor, crest_factor, impulse_factor,
                    clearance_factor, shape_factor, slope, r_value
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_wavelet_features(self, data):
        """
        提取小波特征
        
        Args:
            data: 输入数据 (batch_size, seq_len, channels)
            
        Returns:
            numpy.ndarray: 小波特征
        """
        try:
            import pywt
        except ImportError:
            print("警告: 未安装pywt库，跳过小波特征提取")
            return np.zeros((data.shape[0], data.shape[2] * 8))  # 返回零特征
        
        features = []
        
        for i in range(data.shape[0]):  # 遍历每个样本
            sample_features = []
            
            for j in range(data.shape[2]):  # 遍历每个通道
                channel_data = data[i, :, j]
                
                # 小波分解
                coeffs = pywt.wavedec(channel_data, 'db4', level=3)
                
                # 提取每层系数的统计特征
                for coeff in coeffs:
                    if len(coeff) > 0:
                        mean_coeff = np.mean(coeff)
                        std_coeff = np.std(coeff)
                        sample_features.extend([mean_coeff, std_coeff])
                    else:
                        sample_features.extend([0, 0])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def extract_all_features(self, data):
        """
        提取所有特征
        
        Args:
            data: 输入数据 (batch_size, seq_len, channels)
            
        Returns:
            numpy.ndarray: 所有特征的组合
        """
        print("提取统计特征...")
        stat_features = self.extract_statistical_features(data)
        
        print("提取频域特征...")
        freq_features = self.extract_frequency_features(data)
        
        print("提取时域特征...")
        time_features = self.extract_time_domain_features(data)
        
        print("提取小波特征...")
        wavelet_features = self.extract_wavelet_features(data)
        
        # 合并所有特征
        all_features = np.concatenate([
            stat_features, freq_features, time_features, wavelet_features
        ], axis=1)
        
        print(f"特征维度: {all_features.shape}")
        return all_features
    
    def fit_scaler(self, features):
        """拟合标准化器"""
        self.scaler.fit(features)
    
    def transform_features(self, features):
        """标准化特征"""
        return self.scaler.transform(features)
    
    def fit_transform_features(self, features):
        """拟合并标准化特征"""
        return self.scaler.fit_transform(features)


class EnhancedSignalDataset:
    """增强的信号数据集，包含原始数据和提取的特征"""
    
    def __init__(self, X, y=None, feature_extractor=None):
        """
        初始化数据集
        
        Args:
            X: 原始信号数据
            y: 标签
            feature_extractor: 特征提取器
        """
        self.X_raw = X
        self.y = y
        self.feature_extractor = feature_extractor
        
        if feature_extractor is not None:
            print("提取特征...")
            self.X_features = feature_extractor.extract_all_features(X)
        else:
            self.X_features = None
    
    def __len__(self):
        return len(self.X_raw)
    
    def __getitem__(self, idx):
        raw_data = self.X_raw[idx]
        
        if self.X_features is not None:
            features = self.X_features[idx]
            # 将原始数据和特征合并
            combined_data = np.concatenate([
                raw_data.flatten(),  # 展平原始数据
                features
            ])
        else:
            combined_data = raw_data
        
        if self.y is not None:
            return combined_data, self.y[idx]
        else:
            return combined_data, 0  # 测试集没有标签


def create_enhanced_features(X_train, X_test, y_train=None):
    """
    创建增强特征
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        y_train: 训练标签
        
    Returns:
        tuple: (训练特征, 测试特征, 特征提取器)
    """
    print("创建特征提取器...")
    feature_extractor = FeatureExtractor()
    
    # 提取训练集特征
    print("提取训练集特征...")
    train_features = feature_extractor.extract_all_features(X_train)
    
    # 提取测试集特征
    print("提取测试集特征...")
    test_features = feature_extractor.extract_all_features(X_test)
    
    # 标准化特征
    print("标准化特征...")
    train_features_scaled = feature_extractor.fit_transform_features(train_features)
    test_features_scaled = feature_extractor.transform_features(test_features)
    
    # 合并原始数据和特征
    X_train_enhanced = np.concatenate([
        X_train.reshape(X_train.shape[0], -1),  # 展平原始数据
        train_features_scaled
    ], axis=1)
    
    X_test_enhanced = np.concatenate([
        X_test.reshape(X_test.shape[0], -1),  # 展平原始数据
        test_features_scaled
    ], axis=1)
    
    print(f"增强后的训练数据形状: {X_train_enhanced.shape}")
    print(f"增强后的测试数据形状: {X_test_enhanced.shape}")
    
    return X_train_enhanced, X_test_enhanced, feature_extractor


if __name__ == "__main__":
    # 测试特征提取
    import os
    import sys
    
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.config import *
    
    # 定义数据路径
    TRAIN_DATA_PATH = os.path.join(DATASETS_DIR, 'train.npz')
    TEST_DATA_PATH = os.path.join(DATASETS_DIR, 'test.npz')
    
    # 加载数据
    print("加载数据...")
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    
    X_train, y_train = train_data['samples'], train_data['labels']
    X_test = test_data['samples']
    
    print(f"原始训练数据形状: {X_train.shape}")
    print(f"原始测试数据形状: {X_test.shape}")
    
    # 创建增强特征
    X_train_enhanced, X_test_enhanced, feature_extractor = create_enhanced_features(
        X_train, X_test, y_train
    )
    
    # 保存增强特征
    enhanced_train_path = os.path.join(DATASETS_DIR, 'train_enhanced.npz')
    enhanced_test_path = os.path.join(DATASETS_DIR, 'test_enhanced.npz')
    
    np.savez(enhanced_train_path, X=X_train_enhanced, y=y_train)
    np.savez(enhanced_test_path, X=X_test_enhanced)
    
    print(f"增强特征已保存:")
    print(f"  训练集: {enhanced_train_path}")
    print(f"  测试集: {enhanced_test_path}")