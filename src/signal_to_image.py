"""
信号波形可视化模块
将双通道时序信号数据转换为图像，用于计算机视觉方法进行分类
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from PIL import Image
import io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SignalToImageConverter:
    """信号转图像转换器"""
    
    def __init__(self, image_size=(224, 224), dpi=100, style='default'):
        """
        初始化转换器
        
        Args:
            image_size: 输出图像尺寸 (width, height)
            dpi: 图像分辨率
            style: 绘图风格 ('default', 'clean', 'scientific')
        """
        self.image_size = image_size
        self.dpi = dpi
        self.style = style
        
        # 设置matplotlib样式
        if style == 'clean':
            plt.style.use('seaborn-v0_8-whitegrid')
        elif style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')
        else:
            plt.style.use('default')
    
    def signal_to_image(self, signal_data, title=None, save_path=None):
        """
        将双通道信号转换为图像
        
        Args:
            signal_data: 形状为 (time_steps, 2) 的信号数据
            title: 图像标题
            save_path: 保存路径，如果为None则返回PIL Image对象
            
        Returns:
            PIL Image对象或保存的文件路径
        """
        # 创建图形
        fig_width = self.image_size[0] / self.dpi
        fig_height = self.image_size[1] / self.dpi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=self.dpi)
        
        # 时间轴
        time_steps = np.arange(signal_data.shape[0])
        
        # 绘制通道1
        ax1.plot(time_steps, signal_data[:, 0], 'b-', linewidth=1.0, alpha=0.8)
        ax1.set_ylabel('Channel 1', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(time_steps)-1)
        
        # 绘制通道2
        ax2.plot(time_steps, signal_data[:, 1], 'r-', linewidth=1.0, alpha=0.8)
        ax2.set_ylabel('Channel 2', fontsize=8)
        ax2.set_xlabel('Time Steps', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(time_steps)-1)
        
        # 设置标题
        if title:
            fig.suptitle(title, fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            # 保存到文件
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path
        else:
            # 转换为PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            plt.close()
            return image
    
    def signal_to_spectrogram_image(self, signal_data, title=None, save_path=None):
        """
        将信号转换为频谱图图像
        
        Args:
            signal_data: 形状为 (time_steps, 2) 的信号数据
            title: 图像标题
            save_path: 保存路径
            
        Returns:
            PIL Image对象或保存的文件路径
        """
        fig_width = self.image_size[0] / self.dpi
        fig_height = self.image_size[1] / self.dpi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=self.dpi)
        
        # 计算频谱图
        for i, ax in enumerate([ax1, ax2]):
            Pxx, freqs, bins, im = ax.specgram(signal_data[:, i], NFFT=64, Fs=1.0, 
                                              noverlap=32, cmap='viridis')
            ax.set_ylabel(f'Frequency (Ch{i+1})', fontsize=8)
            if i == 1:
                ax.set_xlabel('Time', fontsize=8)
        
        if title:
            fig.suptitle(title, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            plt.close()
            return image
    
    def signal_to_combined_image(self, signal_data, title=None, save_path=None):
        """
        将信号转换为组合图像（时域+频域）
        
        Args:
            signal_data: 形状为 (time_steps, 2) 的信号数据
            title: 图像标题
            save_path: 保存路径
            
        Returns:
            PIL Image对象或保存的文件路径
        """
        fig_width = self.image_size[0] / self.dpi
        fig_height = self.image_size[1] / self.dpi
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        time_steps = np.arange(signal_data.shape[0])
        
        # 时域信号 - 通道1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_steps, signal_data[:, 0], 'b-', linewidth=0.8)
        ax1.set_title('Channel 1 - Time Domain', fontsize=8)
        ax1.set_ylabel('Amplitude', fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # 时域信号 - 通道2
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_steps, signal_data[:, 1], 'r-', linewidth=0.8)
        ax2.set_title('Channel 2 - Time Domain', fontsize=8)
        ax2.set_ylabel('Amplitude', fontsize=7)
        ax2.set_xlabel('Time Steps', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # 频域分析 - 通道1
        ax3 = fig.add_subplot(gs[0, 1])
        freqs1 = np.fft.fftfreq(len(signal_data), 1.0)
        fft1 = np.abs(np.fft.fft(signal_data[:, 0]))
        ax3.plot(freqs1[:len(freqs1)//2], fft1[:len(fft1)//2], 'b-', linewidth=0.8)
        ax3.set_title('Channel 1 - Frequency Domain', fontsize=8)
        ax3.set_ylabel('Magnitude', fontsize=7)
        ax3.grid(True, alpha=0.3)
        
        # 频域分析 - 通道2
        ax4 = fig.add_subplot(gs[1, 1])
        freqs2 = np.fft.fftfreq(len(signal_data), 1.0)
        fft2 = np.abs(np.fft.fft(signal_data[:, 1]))
        ax4.plot(freqs2[:len(freqs2)//2], fft2[:len(fft2)//2], 'r-', linewidth=0.8)
        ax4.set_title('Channel 2 - Frequency Domain', fontsize=8)
        ax4.set_ylabel('Magnitude', fontsize=7)
        ax4.set_xlabel('Frequency', fontsize=7)
        ax4.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            plt.close()
            return image


class SignalImageDataset(Dataset):
    """信号图像数据集类"""
    
    def __init__(self, signal_data, labels, converter, transform=None, image_type='waveform'):
        """
        初始化数据集
        
        Args:
            signal_data: 信号数据，形状为 (n_samples, time_steps, channels)
            labels: 标签数据
            converter: SignalToImageConverter实例
            transform: 图像变换
            image_type: 图像类型 ('waveform', 'spectrogram', 'combined')
        """
        self.signal_data = signal_data
        self.labels = labels
        self.converter = converter
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.image_type = image_type
    
    def __len__(self):
        return len(self.signal_data)
    
    def __getitem__(self, idx):
        # 获取信号数据
        signal = self.signal_data[idx]
        label = self.labels[idx]
        
        # 转换为图像
        if self.image_type == 'spectrogram':
            image = self.converter.signal_to_spectrogram_image(signal)
        elif self.image_type == 'combined':
            image = self.converter.signal_to_combined_image(signal)
        else:  # waveform
            image = self.converter.signal_to_image(signal)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_sample_images(data_path, output_dir, num_samples_per_class=3):
    """
    创建样本图像用于验证可视化效果
    
    Args:
        data_path: 数据文件路径
        output_dir: 输出目录
        num_samples_per_class: 每个类别生成的样本数
    """
    import os
    
    # 加载数据
    data = np.load(data_path)
    X = data['samples']
    y = data['labels']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建转换器
    converter = SignalToImageConverter(image_size=(224, 224))
    
    # 获取每个类别的样本
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        # 获取该类别的样本索引
        class_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(class_indices, 
                                          min(num_samples_per_class, len(class_indices)), 
                                          replace=False)
        
        for i, idx in enumerate(selected_indices):
            signal = X[idx]
            
            # 生成不同类型的图像
            for img_type in ['waveform', 'spectrogram', 'combined']:
                filename = f"class_{label}_sample_{i}_{img_type}.png"
                filepath = os.path.join(output_dir, filename)
                
                if img_type == 'waveform':
                    converter.signal_to_image(signal, 
                                            title=f"Class {label} - Waveform", 
                                            save_path=filepath)
                elif img_type == 'spectrogram':
                    converter.signal_to_spectrogram_image(signal,
                                                        title=f"Class {label} - Spectrogram",
                                                        save_path=filepath)
                else:  # combined
                    converter.signal_to_combined_image(signal,
                                                     title=f"Class {label} - Combined",
                                                     save_path=filepath)
    
    print(f"样本图像已保存到: {output_dir}")


if __name__ == "__main__":
    # 测试代码
    print("测试信号转图像功能...")
    
    # 创建样本图像
    create_sample_images('artifacts/datasets/train.npz', 'artifacts/sample_images')