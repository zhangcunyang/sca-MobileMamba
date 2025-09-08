import numpy as np, torch
from torch.utils.data import Dataset

class TraceDatasetHighOrder(Dataset):
    """
    生成一阶 (x) + 二阶中心化 (x-μ)² 特征。
    通道数 = 2；shape = (B, 2, T)
    """
    def __init__(self, traces, labels,
                 normalize=True,
                 training=True,
                 noise_std=0.0):
        self.X = np.array(traces, dtype=np.float32)
        self.Y = np.array(labels,  dtype=np.int64)
        self.training  = training
        self.noise_std = noise_std

        # 全局 z-score（对二阶特征也有利）
        if normalize:
            μ, σ = self.X.mean(), self.X.std()
            σ = σ if σ > 1e-6 else 1.0
            self.X = (self.X - μ) / σ

        # 预计算二阶中心化
        x_centered = self.X - self.X.mean(axis=1, keepdims=True)
        x2 = x_centered ** 2                     # (N, T)
        x2 *= 0.5                                # 缩放因子，平衡两通道量级

        # 组合为 (N, 2, T)
        self.X = np.stack([self.X, x2], axis=1)
        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x, y
