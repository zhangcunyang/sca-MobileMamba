# experiments/run_ascad_mamba.py  ▶ 高阶二通道版
import os, time, torch, h5py, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 项目内部 ======
from ascad_loader_fixed import load_ascad_fixed_key
from trace_dataset_highorder import TraceDatasetHighOrder as TraceDataset
from mobile_mamba import MobileMambaSCA
from train import train_one_epoch
from evaluate import evaluate, evaluate_per_class, evaluate_guessing_entropy

# ---------- 配置 ----------
TARGET_BYTE  = 0
WINDOW       = (40000, 50000)   # 修复窗口大小到官方建议范围
EPOCHS       = 40
BATCH_SIZE   = 128
LR           = 1e-3
NOISE_STD    = 0.0
LOG_DIR      = "logs_highorder"
SAVE_PATH    = "checkpoints/mamba_highorder.pth"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- 数据 ----------
(train_X, train_Y), (test_X, test_Y) = load_ascad_fixed_key(
    window=WINDOW, target_byte=TARGET_BYTE)

train_ds = TraceDataset(train_X, train_Y, normalize=True,
                        training=True,  noise_std=NOISE_STD)
test_ds  = TraceDataset(test_X,  test_Y, normalize=True,
                        training=False, noise_std=0.0)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ---------- 模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MobileMambaSCA(input_len=train_X.shape[1],
                        in_ch=2,          # ★ 二阶 → 2 通道
                        dim=128,
                        num_blocks=6).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- 训练 ----------
train_losses, test_accs = [], []
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
    acc  = evaluate(model, test_dl, device)
    scheduler.step()  # 更新学习率
    train_losses.append(loss); test_accs.append(acc)
    print(f"  • train_loss={loss:.4f}   test_acc={acc:.4f}   lr={scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"\n✅ Model saved to {SAVE_PATH}")

# ---------- 曲线 ----------
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Loss"); plt.xlabel("epoch"); plt.grid(); plt.legend()
plt.tight_layout(); plt.savefig(f"{LOG_DIR}/loss.png"); plt.close()

plt.figure(figsize=(8,4))
plt.plot(test_accs, label="Accuracy"); plt.xlabel("epoch"); plt.grid(); plt.legend()
plt.tight_layout(); plt.savefig(f"{LOG_DIR}/acc.png"); plt.close()

# ---------- per-class 报告 ----------
_ = evaluate_per_class(model, test_dl, device)

# ---------- Guessing-Entropy ----------
ATT_NUM = 1000
attack_ds = TraceDataset(test_X[:ATT_NUM], test_Y[:ATT_NUM],
                         normalize=True, training=False)
attack_dl = DataLoader(attack_ds, batch_size=64)

with h5py.File("./datasets/raw/ASCAD_fixed_key.h5", "r") as f:
    pts   = f["metadata"][50000:50000+ATT_NUM]["plaintext"][:, TARGET_BYTE]
    trueK = int(f["metadata"][50000]["key"][TARGET_BYTE])

ge_curve = evaluate_guessing_entropy(model, attack_dl, pts, trueK,
                                     device, max_traces=ATT_NUM)

plt.figure(figsize=(6,4))
plt.plot(ge_curve, marker="o"); plt.grid()
plt.title("Guessing Entropy (byte-0, 2-order)"); plt.xlabel("#traces"); plt.ylabel("GE")
plt.tight_layout(); plt.savefig(f"{LOG_DIR}/ge_curve.png"); plt.close()

print(f"\nAll plots saved to {LOG_DIR}/")
