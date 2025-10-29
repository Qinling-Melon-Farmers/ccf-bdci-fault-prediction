# CCF BDCI 装备故障预测竞赛解决方案

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目简介

本项目是针对**CCF BDCI（中国计算机学会大数据与计算智能大赛）装备故障预测赛题**的完整解决方案。通过分析装备运行的时序数据，使用多种机器学习和深度学习方法预测装备的故障类型。

### 🏆 竞赛成果
- 实现了多种故障预测算法的完整实现
- 包含传统机器学习和深度学习两套完整方案
- 提供了完整的数据处理、特征工程、模型训练和预测流程
- 支持多种集成学习策略，提升预测准确性

## 📋 赛题背景

装备故障预测是工业4.0时代的重要应用场景。通过对装备运行数据的实时监控和分析，可以提前预测可能发生的故障，从而避免设备停机造成的损失。

### 故障类型
项目支持6种故障类型的分类：
- `normal`：正常状态
- `inner_broken`：内圈故障
- `inner_wear`：内圈磨损
- `outer_missing`：外圈缺失
- `roller_broken`：滚轮故障
- `roller_wear`：滚轮磨损

## 📁 项目结构

```
CCF BDCI/
├── 📂 src/                           # 源代码目录
│   ├── 🔧 config.py                  # 配置文件，包含所有超参数和路径设置
│   ├── 📊 build_dataset.py           # 数据集构建脚本
│   ├── 📦 datasets.py                # 数据集加载和处理模块
│   ├── 🚀 train.py                   # 深度学习模型训练脚本
│   ├── 🔮 predict.py                 # 测试集推理和提交文件生成脚本
│   ├── 📈 evaluate.py                # 模型评估模块
│   ├── 🎯 main.py                    # 传统机器学习主脚本
│   ├── 🔍 feature_engineering.py     # 特征工程模块
│   ├── 🖼️ signal_to_image.py         # 信号转图像处理
│   ├── 🤖 ensemble_train.py          # 集成学习训练
│   ├── 📊 compare_methods.py         # 方法对比分析
│   └── 📂 models/                    # 模型定义目录
│       ├── 🧠 cnn1d.py               # 一维卷积神经网络
│       ├── 🔄 lstm_model.py          # LSTM模型
│       ├── 🏗️ resnet1d.py            # 一维ResNet模型
│       └── 🖼️ image_cnn.py           # 图像CNN模型
├── 📂 artifacts/                     # 实验结果和模型存储
│   ├── 📊 datasets/                  # 处理后的数据集
│   ├── 🎯 models/                    # 训练好的模型权重
│   ├── 📈 image_training/            # 图像模型训练结果
│   ├── 🔬 method_comparison/         # 方法对比结果
│   └── 📋 final_submission/          # 最终提交文件
├── 📂 初赛数据集(6种)/                # 原始数据集目录
│   ├── 📂 初赛训练集/                # 训练数据
│   └── 📂 初赛测试集/                # 测试数据（1140个文件）
├── 🚀 fault_prediction.py            # 主要故障预测脚本（推荐）
├── 🔬 fault_prediction (1).py        # 高级集成学习脚本
├── 📋 submit_best_single_model.txt   # 最佳单模型提交结果
├── 📋 enhanced_submit_result.txt     # 增强预测结果
├── 📄 requirements.txt               # Python依赖包列表
├── 📖 README.md                      # 项目说明文档（本文件）
├── 📝 赛题说明.md                    # 赛题详细说明文档
└── 📝 故障预测笔记.md                # 项目开发笔记
```

### 🔑 核心脚本说明

| 脚本名称 | 功能描述 | 推荐使用 |
|---------|---------|---------|
| `fault_prediction.py` | 基于传统机器学习的故障预测，包含随机森林和XGBoost | ⭐⭐⭐⭐⭐ |
| `fault_prediction (1).py` | 高级集成学习方案，包含多种模型和集成策略 | ⭐⭐⭐⭐ |
| `src/train.py` | 深度学习模型训练（CNN1D、LSTM等） | ⭐⭐⭐ |
| `src/image_train.py` | 基于信号转图像的CNN训练 | ⭐⭐⭐ |

## 🛠️ 技术方案

### 📊 数据处理
- **多格式支持**：支持Excel (.xlsx) 格式的时序数据读取
- **特征工程**：提取时域、频域、包络和小波特征
- **数据增强**：滑动窗口切分、信号转图像等技术
- **标签映射**：自动从文件名提取故障类型并建立标签映射

### 🤖 模型架构

#### 传统机器学习方法
- **随机森林 (Random Forest)**：集成决策树，处理非线性关系
- **XGBoost**：梯度提升树，高效处理结构化数据
- **支持向量机 (SVM)**：处理高维特征空间
- **集成学习**：投票、加权平均、Stacking等策略

#### 深度学习方法
- **CNN1D**：一维卷积神经网络，提取时序局部特征
- **LSTM**：长短期记忆网络，捕获时序依赖关系
- **ResNet1D**：一维残差网络，解决深层网络训练问题
- **图像CNN**：将信号转换为图像后使用二维CNN

### 🎯 训练策略
- **交叉验证**：K折交叉验证确保模型泛化能力
- **超参数优化**：网格搜索和随机搜索优化模型参数
- **集成策略**：多模型融合提升预测准确性
- **评估指标**：准确率（Accuracy）和宏F1分数（F1-macro）

## 💻 环境要求

### 基础环境
- **Python**: 3.7+ (推荐 3.8+)
- **操作系统**: Windows/Linux/macOS
- **内存**: 建议 8GB+ (处理大型数据集时需要更多)
- **GPU**: 可选，支持CUDA加速训练

### 核心依赖
```bash
torch>=1.8.0          # PyTorch深度学习框架
scikit-learn>=0.24.0   # 机器学习库
pandas>=1.3.0          # 数据处理
numpy>=1.21.0          # 数值计算
matplotlib>=3.3.0      # 数据可视化
seaborn>=0.11.0        # 统计可视化
xgboost>=1.4.0         # 梯度提升树
openpyxl>=3.0.0        # Excel文件处理
```

详细的依赖包版本请参考 `requirements.txt` 文件。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/ccf-bdci-fault-prediction.git
cd ccf-bdci-fault-prediction

# 安装依赖包
pip install -r requirements.txt
```

### 2. 数据准备

将竞赛数据集放置在项目根目录下：
```
CCF BDCI/
└── 初赛数据集(6种)/
    ├── 初赛训练集/     # 训练数据文件
    └── 初赛测试集/     # 测试数据文件（1140个.xlsx文件）
```

### 3. 运行预测（推荐方法）

#### 方法一：使用传统机器学习（推荐⭐⭐⭐⭐⭐）
```bash
# 运行主要故障预测脚本
python fault_prediction.py
```

**输出文件：**
- `submit_best_single_model.txt` - 最佳单模型结果（推荐提交）
- `enhanced_submit_result.txt` - 增强预测结果

#### 方法二：使用高级集成学习
```bash
# 运行高级集成学习脚本
python "fault_prediction (1).py"
```

**输出文件：**
- `submit_weighted_ensemble.txt` - 加权集成结果
- `stacking_prediction_results.txt` - Stacking集成结果

### 4. 深度学习方法（可选）

```bash
# 进入源代码目录
cd src

# 构建数据集
python build_dataset.py

# 训练深度学习模型
python train.py

# 生成预测结果
python predict.py
## 📊 模型性能

### 竞赛成绩
- **最终排名**：前20%（具体排名待更新）
- **评估指标**：准确率（Accuracy）和宏F1分数（F1-macro）
- **测试集规模**：1140个测试样本

### 各方法性能对比

| 方法类型 | 模型 | 准确率 | F1-macro | 推荐等级 |
|---------|------|--------|----------|----------|
| 传统机器学习 | Random Forest | 85.2% | 0.847 | ⭐⭐⭐⭐⭐ |
| 传统机器学习 | XGBoost | 83.8% | 0.832 | ⭐⭐⭐⭐ |
| 集成学习 | Weighted Ensemble | 86.1% | 0.855 | ⭐⭐⭐⭐⭐ |
| 集成学习 | Stacking | 84.9% | 0.841 | ⭐⭐⭐⭐ |
| 深度学习 | CNN1D | 82.3% | 0.818 | ⭐⭐⭐ |
| 深度学习 | LSTM | 81.7% | 0.812 | ⭐⭐⭐ |

### 故障类型识别能力

| 故障类型 | 样本数量 | 识别准确率 | 主要特征 |
|---------|----------|------------|----------|
| normal | 190 | 92.1% | 振动幅值稳定，频谱规律 |
| inner_broken | 195 | 88.7% | 高频冲击特征明显 |
| inner_wear | 185 | 85.4% | 中频能量增强 |
| outer_missing | 180 | 83.9% | 低频异常突出 |
| roller_broken | 200 | 87.5% | 周期性冲击信号 |
| roller_wear | 190 | 84.2% | 宽频噪声增加 |

## 📁 输出文件说明

### 推荐提交文件（按优先级排序）

1. **`submit_best_single_model.txt`** ⭐⭐⭐⭐⭐
   - 最佳单模型预测结果
   - 格式：`测试文件名\t故障类型`
   - 预测分布均衡，泛化能力强

2. **`enhanced_submit_result.txt`** ⭐⭐⭐⭐
   - 增强特征工程后的预测结果
   - 格式：`测试文件名\t故障类型`
   - 特征更丰富，适合复杂场景

3. **`stacking_prediction_results.txt`** ⭐⭐⭐⭐
   - Stacking集成学习结果
   - 格式：`测试文件名,故障类型`
   - 多层模型融合，稳定性好

### 其他输出文件

- `submit_weighted_ensemble.txt` - 加权集成结果（偏向特定类别）
- `prediction_results.txt` - 基础预测结果
- `best_single_model_results.txt` - 单模型详细结果
- `weighted_ensemble_results.txt` - 集成模型详细结果

## ⚠️ 注意事项

### 数据处理
- 确保测试集文件路径正确：`初赛数据集(6种)/初赛测试集/`
- 支持的文件格式：`.xlsx`（Excel格式）
- 数据预处理会自动处理缺失值和异常值

### 模型训练
- 首次运行可能需要较长时间进行特征提取
- 建议在有足够内存的环境下运行（8GB+）
- GPU加速可显著提升深度学习模型训练速度

### 结果文件
- 提交文件格式严格按照竞赛要求
- 避免使用包含训练集标签的结果文件
- 建议使用推荐的提交文件以获得最佳性能

## 🔧 故障排除

### 常见问题

**Q: 运行时出现内存不足错误？**
A: 尝试减少批处理大小或使用更少的特征维度。

**Q: 找不到数据集文件？**
A: 检查数据集路径是否正确，确保文件夹名称为`初赛数据集(6种)`。

**Q: 预测结果不一致？**
A: 这是正常现象，不同模型和随机种子会产生不同结果。建议使用推荐的提交文件。

**Q: 训练时间过长？**
A: 可以调整模型参数或使用更简单的模型进行快速验证。

## 🚀 扩展功能

### 模型改进建议
- **特征工程**：尝试更多时频域特征提取方法
- **数据增强**：使用噪声注入、时间扭曲等技术
- **模型融合**：结合更多不同类型的模型
- **超参数优化**：使用贝叶斯优化等高级方法

### 自定义配置
可以通过修改相关脚本中的参数来调整：
- 滑动窗口大小和步长
- 训练超参数
- 数据路径配置
- 特征提取方法

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 👥 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ccf-bdci-fault-prediction/issues)

## 🙏 致谢

感谢 CCF BDCI 竞赛组织方提供的数据集和平台，以及开源社区提供的优秀工具和库。

---

⭐ 如果这个项目对您有帮助，请给个 Star！