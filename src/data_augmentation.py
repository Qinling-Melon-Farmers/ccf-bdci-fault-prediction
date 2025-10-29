"""
数据增强模块
实现多种数据增强策略来增加训练数据的多样性
"""
import numpy as np
import torch
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


class SignalAugmenter:
    """信号数据增强器"""
    
    def __init__(self, noise_factor=0.01, time_shift_factor=0.1, 
                 scale_factor=0.1, rotation_angle=5):
        """
        初始化数据增强器
        
        Args:
            noise_factor: 噪声强度
            time_shift_factor: 时间偏移因子
            scale_factor: 缩放因子
            rotation_angle: 旋转角度
        """
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle
    
    def add_gaussian_noise(self, data):
        """
        添加高斯噪声
        
        Args:
            data: 输入数据 (seq_len, channels)
            
        Returns:
            numpy.ndarray: 添加噪声后的数据
        """
        noise = np.random.normal(0, self.noise_factor, data.shape)
        return data + noise
    
    def time_shift(self, data):
        """
        时间偏移
        
        Args:
            data: 输入数据 (seq_len, channels)
            
        Returns:
            numpy.ndarray: 时间偏移后的数据
        """
        seq_len = data.shape[0]
        shift = int(seq_len * self.time_shift_factor * (np.random.random() - 0.5))
        
        if shift > 0:
            # 向右偏移
            shifted_data = np.zeros_like(data)
            shifted_data[shift:] = data[:-shift]
            shifted_data[:shift] = data[-shift:]  # 循环填充
        elif shift < 0:
            # 向左偏移
            shifted_data = np.zeros_like(data)
            shifted_data[:shift] = data[-shift:]
            shifted_data[shift:] = data[:-shift]  # 循环填充
        else:
            shifted_data = data.copy()
        
        return shifted_data
    
    def amplitude_scaling(self, data):
        """
        幅度缩放
        
        Args:
            data: 输入数据 (seq_len, channels)
            
        Returns:
            numpy.ndarray: 缩放后的数据
        """
        scale = 1 + self.scale_factor * (np.random.random() - 0.5)
        return data * scale
    
    def frequency_masking(self, data, mask_ratio=0.1):
        """
        频域掩码
        
        Args:
            data: 输入数据 (seq_len, channels)
            mask_ratio: 掩码比例
            
        Returns:
            numpy.ndarray: 频域掩码后的数据
        """
        augmented_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):  # 对每个通道处理
            channel_data = data[:, i]
            
            # FFT变换
            fft_data = np.fft.fft(channel_data)
            
            # 随机掩码频率成分
            mask_size = int(len(fft_data) * mask_ratio)
            mask_start = np.random.randint(0, len(fft_data) - mask_size)
            fft_data[mask_start:mask_start + mask_size] = 0
            
            # 逆FFT变换
            augmented_data[:, i] = np.real(np.fft.ifft(fft_data))
        
        return augmented_data
    
    def time_masking(self, data, mask_ratio=0.1):
        """
        时域掩码
        
        Args:
            data: 输入数据 (seq_len, channels)
            mask_ratio: 掩码比例
            
        Returns:
            numpy.ndarray: 时域掩码后的数据
        """
        augmented_data = data.copy()
        seq_len = data.shape[0]
        
        # 随机掩码时间段
        mask_size = int(seq_len * mask_ratio)
        mask_start = np.random.randint(0, seq_len - mask_size)
        augmented_data[mask_start:mask_start + mask_size] = 0
        
        return augmented_data
    
    def add_trend(self, data):
        """
        添加趋势
        
        Args:
            data: 输入数据 (seq_len, channels)
            
        Returns:
            numpy.ndarray: 添加趋势后的数据
        """
        seq_len = data.shape[0]
        trend_strength = 0.01 * (np.random.random() - 0.5)
        
        # 创建线性趋势
        trend = np.linspace(0, trend_strength, seq_len).reshape(-1, 1)
        trend = np.tile(trend, (1, data.shape[1]))
        
        return data + trend
    
    def elastic_transform(self, data, alpha=1, sigma=0.1):
        """
        弹性变换
        
        Args:
            data: 输入数据 (seq_len, channels)
            alpha: 变换强度
            sigma: 高斯核标准差
            
        Returns:
            numpy.ndarray: 弹性变换后的数据
        """
        seq_len = data.shape[0]
        
        # 生成随机位移场
        dx = np.random.randn(seq_len) * alpha
        dx = gaussian_filter1d(dx, sigma, mode='constant', cval=0)
        
        # 应用位移
        indices = np.arange(seq_len) + dx
        indices = np.clip(indices, 0, seq_len - 1)
        
        augmented_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            augmented_data[:, i] = np.interp(np.arange(seq_len), indices, data[:, i])
        
        return augmented_data
    
    def mixup(self, data1, data2, alpha=0.2):
        """
        Mixup数据增强
        
        Args:
            data1: 第一个样本 (seq_len, channels)
            data2: 第二个样本 (seq_len, channels)
            alpha: mixup参数
            
        Returns:
            tuple: (混合数据, 混合权重)
        """
        lam = np.random.beta(alpha, alpha)
        mixed_data = lam * data1 + (1 - lam) * data2
        return mixed_data, lam
    
    def cutmix(self, data1, data2, alpha=1.0):
        """
        CutMix数据增强
        
        Args:
            data1: 第一个样本 (seq_len, channels)
            data2: 第二个样本 (seq_len, channels)
            alpha: cutmix参数
            
        Returns:
            tuple: (混合数据, 混合权重)
        """
        seq_len = data1.shape[0]
        lam = np.random.beta(alpha, alpha)
        
        # 计算切割区域
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len
        
        # 创建混合数据
        mixed_data = data1.copy()
        mixed_data[cut_start:cut_end] = data2[cut_start:cut_end]
        
        # 计算实际混合比例
        actual_lam = 1 - cut_len / seq_len
        
        return mixed_data, actual_lam
    
    def augment_single(self, data, augment_type='random'):
        """
        对单个样本进行数据增强
        
        Args:
            data: 输入数据 (seq_len, channels)
            augment_type: 增强类型
            
        Returns:
            numpy.ndarray: 增强后的数据
        """
        if augment_type == 'random':
            # 随机选择增强方法
            methods = [
                self.add_gaussian_noise,
                self.time_shift,
                self.amplitude_scaling,
                self.frequency_masking,
                self.time_masking,
                self.add_trend,
                self.elastic_transform
            ]
            method = np.random.choice(methods)
            return method(data)
        
        elif augment_type == 'noise':
            return self.add_gaussian_noise(data)
        elif augment_type == 'shift':
            return self.time_shift(data)
        elif augment_type == 'scale':
            return self.amplitude_scaling(data)
        elif augment_type == 'freq_mask':
            return self.frequency_masking(data)
        elif augment_type == 'time_mask':
            return self.time_masking(data)
        elif augment_type == 'trend':
            return self.add_trend(data)
        elif augment_type == 'elastic':
            return self.elastic_transform(data)
        else:
            return data


class AugmentedDataset:
    """增强数据集"""
    
    def __init__(self, X, y, augmenter, augment_ratio=2.0, use_mixup=True):
        """
        初始化增强数据集
        
        Args:
            X: 原始数据 (batch_size, seq_len, channels)
            y: 原始标签
            augmenter: 数据增强器
            augment_ratio: 增强比例
            use_mixup: 是否使用mixup
        """
        self.X_original = X
        self.y_original = y
        self.augmenter = augmenter
        self.augment_ratio = augment_ratio
        self.use_mixup = use_mixup
        
        # 生成增强数据
        self.X_augmented, self.y_augmented = self._generate_augmented_data()
        
        # 合并原始数据和增强数据
        self.X = np.concatenate([self.X_original, self.X_augmented], axis=0)
        self.y = np.concatenate([self.y_original, self.y_augmented], axis=0)
        
        # 打乱数据
        self.X, self.y = shuffle(self.X, self.y, random_state=42)
    
    def _generate_augmented_data(self):
        """生成增强数据"""
        n_samples = len(self.X_original)
        n_augmented = int(n_samples * self.augment_ratio)
        
        X_aug = []
        y_aug = []
        
        print(f"生成 {n_augmented} 个增强样本...")
        
        for i in range(n_augmented):
            # 随机选择一个原始样本
            idx = np.random.randint(0, n_samples)
            sample = self.X_original[idx]
            label = self.y_original[idx]
            
            if self.use_mixup and np.random.random() < 0.3:  # 30%概率使用mixup
                # 选择另一个样本进行mixup
                idx2 = np.random.randint(0, n_samples)
                sample2 = self.X_original[idx2]
                label2 = self.y_original[idx2]
                
                if np.random.random() < 0.5:
                    # 使用mixup
                    mixed_sample, lam = self.augmenter.mixup(sample, sample2)
                    # 对于分类任务，选择主要标签
                    mixed_label = label if lam > 0.5 else label2
                else:
                    # 使用cutmix
                    mixed_sample, lam = self.augmenter.cutmix(sample, sample2)
                    mixed_label = label if lam > 0.5 else label2
                
                X_aug.append(mixed_sample)
                y_aug.append(mixed_label)
            else:
                # 使用常规增强
                aug_sample = self.augmenter.augment_single(sample, 'random')
                X_aug.append(aug_sample)
                y_aug.append(label)
        
        return np.array(X_aug), np.array(y_aug)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_augmented_dataset(X_train, y_train, augment_ratio=2.0):
    """
    创建增强数据集
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        augment_ratio: 增强比例
        
    Returns:
        tuple: (增强后的X, 增强后的y)
    """
    print("创建数据增强器...")
    augmenter = SignalAugmenter(
        noise_factor=0.01,
        time_shift_factor=0.1,
        scale_factor=0.1,
        rotation_angle=5
    )
    
    print("生成增强数据集...")
    augmented_dataset = AugmentedDataset(
        X_train, y_train, augmenter, 
        augment_ratio=augment_ratio, 
        use_mixup=True
    )
    
    print(f"原始数据集大小: {len(X_train)}")
    print(f"增强后数据集大小: {len(augmented_dataset)}")
    
    return augmented_dataset.X, augmented_dataset.y


if __name__ == "__main__":
    # 测试数据增强
    import os
    import sys
    
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.config import *
    
    # 定义数据路径
    TRAIN_DATA_PATH = os.path.join(DATASETS_DIR, 'train.npz')
    
    # 加载数据
    print("加载数据...")
    train_data = np.load(TRAIN_DATA_PATH)
    X_train, y_train = train_data['samples'], train_data['labels']
    
    print(f"原始训练数据形状: {X_train.shape}")
    print(f"原始标签形状: {y_train.shape}")
    
    # 创建增强数据集
    X_augmented, y_augmented = create_augmented_dataset(
        X_train, y_train, augment_ratio=1.0  # 增加1倍数据
    )
    
    # 保存增强数据
    augmented_path = os.path.join(DATASETS_DIR, 'train_augmented.npz')
    np.savez(augmented_path, samples=X_augmented, labels=y_augmented)
    
    print(f"增强数据已保存到: {augmented_path}")
    
    # 统计类别分布
    unique_original, counts_original = np.unique(y_train, return_counts=True)
    unique_augmented, counts_augmented = np.unique(y_augmented, return_counts=True)
    
    print("\n原始数据类别分布:")
    for cls, count in zip(unique_original, counts_original):
        print(f"  类别 {cls}: {count} 个样本")
    
    print("\n增强后数据类别分布:")
    for cls, count in zip(unique_augmented, counts_augmented):
        print(f"  类别 {cls}: {count} 个样本")