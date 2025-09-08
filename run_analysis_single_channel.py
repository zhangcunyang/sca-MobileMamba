# 使用现有单通道模型进行增强分析
import os, time, torch, h5py, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 项目内部 ======
from ascad_loader_fixed import load_ascad_fixed_key
from trace_dataset_highorder import TraceDatasetHighOrder as TraceDataset
from mobile_mamba import MobileMambaSCA
from evaluate import evaluate, evaluate_per_class, evaluate_guessing_entropy
from enhanced_evaluate import EnhancedEvaluator

# ---------- 配置 ----------
TARGET_BYTE  = 0
WINDOW       = (6471, 7271)   # 来自高阶 SNR 扫描
BATCH_SIZE   = 128
LOG_DIR      = "logs_enhanced"
EXISTING_MODEL_PATH = "checkpoints/mamba_model.pth"  # 使用现有的单通道模型
ANALYSIS_DIR = "analysis_results"

# 创建目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

print("=" * 60)
print("使用现有单通道模型进行增强分析")
print("=" * 60)

# ---------- 数据加载 ----------
print("加载ASCAD数据集...")
(train_X, train_Y), (test_X, test_Y) = load_ascad_fixed_key(
    window=WINDOW, target_byte=TARGET_BYTE)

print(f"训练集: {train_X.shape}, 测试集: {test_X.shape}")

# 创建单通道数据集（使用原始的一阶特征）
class SingleChannelDataset:
    """单通道数据集，用于兼容现有模型"""
    def __init__(self, traces, labels, normalize=True, training=False, noise_std=0.0):
        self.X = np.array(traces, dtype=np.float32)
        self.Y = np.array(labels, dtype=np.int64)
        self.training = training
        self.noise_std = noise_std
        
        # 全局 z-score 标准化
        if normalize:
            μ, σ = self.X.mean(), self.X.std()
            σ = σ if σ > 1e-6 else 1.0
            self.X = (self.X - μ) / σ
        
        # 转换为单通道格式 (N, 1, T)
        self.X = torch.from_numpy(self.X).unsqueeze(1)  # (N, 1, T)
        self.Y = torch.from_numpy(self.Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x, y

# 创建单通道测试数据集
test_ds = SingleChannelDataset(test_X, test_Y, normalize=True, training=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------- 模型加载 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建单通道模型（带bias以兼容现有模型）
class CompatibleMobileMambaSCA(MobileMambaSCA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加bias以兼容现有模型
        self.embed_33 = nn.Conv1d(1, 64, kernel_size=33, padding=16, stride=2, bias=True)
        self.embed_51 = nn.Conv1d(1, 64, kernel_size=51, padding=25, stride=2, bias=True)

model = CompatibleMobileMambaSCA(input_len=train_X.shape[1],
                                in_ch=1,          # 单通道
                                dim=128,
                                num_blocks=6).to(device)

# 加载现有模型
if os.path.exists(EXISTING_MODEL_PATH):
    print(f"加载现有模型: {EXISTING_MODEL_PATH}")
    model.load_state_dict(torch.load(EXISTING_MODEL_PATH, map_location=device))
    print("✅ 模型加载成功")
else:
    print(f"❌ 找不到模型文件: {EXISTING_MODEL_PATH}")
    exit(1)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# ---------- 基础评估 ----------
print("\n" + "=" * 40)
print("基础评估")
print("=" * 40)

# 测试准确率
print("计算测试准确率...")
test_acc = evaluate(model, test_dl, device)
print(f"测试准确率: {test_acc:.4f}")

# 每类别分类报告
print("生成每类别分类报告...")
_ = evaluate_per_class(model, test_dl, device)

# ---------- 增强分析 ----------
print("\n" + "=" * 40)
print("增强分析 - 猜测熵和t-SNE可视化")
print("=" * 40)

# 准备攻击数据
ATT_NUM = 2000  # 攻击样本数量
attack_ds = SingleChannelDataset(test_X[:ATT_NUM], test_Y[:ATT_NUM],
                                normalize=True, training=False)
attack_dl = DataLoader(attack_ds, batch_size=64)

# 获取明文和真实密钥
with h5py.File("./datasets/raw/ASCAD_fixed_key.h5", "r") as f:
    pts   = f["metadata"][50000:50000+ATT_NUM]["plaintext"][:, TARGET_BYTE]
    trueK = int(f["metadata"][50000]["key"][TARGET_BYTE])

print(f"真实密钥字节 {TARGET_BYTE}: {trueK}")
print(f"攻击样本数量: {ATT_NUM}")

# 创建增强评估器
evaluator = EnhancedEvaluator(model, device, num_classes=256)

# 执行综合分析
print("开始综合分析...")
class_ge, poor_classes = evaluator.comprehensive_analysis(
    attack_dl, pts, trueK, 
    target_byte=TARGET_BYTE, 
    max_traces=ATT_NUM,
    ge_threshold=30,  # 猜测熵阈值
    save_dir=ANALYSIS_DIR
)

# ---------- 详细分析报告 ----------
print("\n" + "=" * 40)
print("详细分析报告")
print("=" * 40)

print(f"总共分析了 {len(class_ge)} 个类别")
print(f"发现 {len(poor_classes)} 个分不出来的类别")

if poor_classes:
    print("\n最差的10个类别:")
    for i, (class_id, ge) in enumerate(poor_classes[:10]):
        print(f"  {i+1:2d}. 类别 {class_id:3d}: 平均猜测熵 = {ge:6.2f}")

# 计算统计信息
ge_values = list(class_ge.values())
print(f"\n猜测熵统计:")
print(f"  最小值: {min(ge_values):.2f}")
print(f"  最大值: {max(ge_values):.2f}")
print(f"  平均值: {np.mean(ge_values):.2f}")
print(f"  中位数: {np.median(ge_values):.2f}")
print(f"  标准差: {np.std(ge_values):.2f}")

# 绘制猜测熵分布
plt.figure(figsize=(10, 6))
plt.hist(ge_values, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(ge_values), color='red', linestyle='--', 
           label=f'平均值: {np.mean(ge_values):.2f}')
plt.axvline(np.median(ge_values), color='green', linestyle='--', 
           label=f'中位数: {np.median(ge_values):.2f}')
plt.xlabel('猜测熵')
plt.ylabel('类别数量')
plt.title('类别猜测熵分布 (单通道模型)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/ge_distribution_single_channel.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------- 传统猜测熵曲线 ----------
print("\n计算传统猜测熵曲线...")
ge_curve = evaluate_guessing_entropy(model, attack_dl, pts, trueK,
                                     device, max_traces=ATT_NUM)

plt.figure(figsize=(8, 5))
plt.plot(ge_curve, marker="o", markersize=2, linewidth=1)
plt.grid(True, alpha=0.3)
plt.title(f"传统猜测熵曲线 (字节 {TARGET_BYTE}, 单通道)")
plt.xlabel("攻击轨迹数量")
plt.ylabel("猜测熵")
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/traditional_ge_curve_single_channel.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"最终猜测熵: {ge_curve[-1] if ge_curve else 'N/A'}")

# ---------- 总结 ----------
print("\n" + "=" * 60)
print("分析完成总结")
print("=" * 60)
print(f"✅ 使用现有单通道模型: {EXISTING_MODEL_PATH}")
print(f"✅ 测试准确率: {test_acc:.4f}")
print(f"✅ 分析结果保存在: {ANALYSIS_DIR}/")
print(f"✅ 发现 {len(poor_classes)} 个分不出来的类别")
if ge_curve:
    print(f"✅ 传统猜测熵: {ge_curve[-1]}")
print("\n生成的文件:")
print(f"  - {LOG_DIR}/traditional_ge_curve_single_channel.png (传统猜测熵曲线)")
print(f"  - {ANALYSIS_DIR}/class_guessing_entropy.png (类别猜测熵)")
print(f"  - {ANALYSIS_DIR}/ge_distribution_single_channel.png (猜测熵分布)")
print(f"  - {ANALYSIS_DIR}/tsne_worst_classes.png (t-SNE可视化)")
print(f"  - {ANALYSIS_DIR}/class_analysis_results.csv (详细结果)")

print("\n🎉 使用现有单通道模型的分析完成！")
