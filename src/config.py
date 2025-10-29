"""
项目配置文件
用于集中管理项目的各种配置参数和路径设置
"""
import os

# ==================== 数据路径配置 ====================
# 数据集根目录
DATA_ROOT = r"C:\Users\ASUS\Desktop\CCF BDCI\初赛数据集(6种)"  # 数据集根目录
TRAIN_DIR = os.path.join(DATA_ROOT, "初赛训练集")  # 训练数据目录
TEST_DIR = os.path.join(DATA_ROOT, "初赛测试集")    # 测试数据目录

# ==================== 滑动窗口配置 ====================
# 滑动窗口大小（无重叠）
WINDOW_SIZE = 500
# 滑动步长（与窗口大小相等，确保无重叠）
STEP_SIZE = 500

# ==================== 训练参数配置 ====================
# 类别数量（初赛阶段为6种故障类型）
NUM_CLASSES = 6
# 批次大小
BATCH_SIZE = 8
# 训练轮数
EPOCHS = 50
# 学习率
LEARNING_RATE = 1e-3

# ==================== 输出路径配置 ====================
# 生成文件的根目录
ARTIFACTS_DIR = os.path.join(r"c:\Users\ASUS\Desktop\CCF BDCI", "artifacts")
# 数据集文件存储目录
DATASETS_DIR = os.path.join(ARTIFACTS_DIR, "datasets")
# 模型文件存储目录
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
# 提交文件路径
SUBMIT_PATH = os.path.join(r"c:\Users\ASUS\Desktop\CCF BDCI", "submit.txt")

# 创建必要的目录
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)