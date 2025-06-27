import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# ① Real-Mamba selective-scan block（保持不变）
# -----------------------------------------------------------
class RealMambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16,
                 conv_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # depth-wise conv for local mixing
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel,
                                 padding=conv_kernel // 2, groups=dim, bias=False)

        # projection to (u, v, gate)
        self.in_proj = nn.Linear(dim, dim * 3, bias=False)

        # diagonal SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Parameter(torch.randn(state_dim))
        self.C = nn.Parameter(torch.randn(state_dim))
        self.D = nn.Parameter(torch.randn(state_dim))

        # output proj
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        residual = x
        B, T, C = x.shape

        # 1) depth-wise conv
        x_local = self.dw_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 2) gated projections
        u, v, gate = self.in_proj(x_local).chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        v = F.silu(v)

        # 3) Selective Scan (diagonal SSM)
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        y = []

        A = torch.tanh(self.A)
        Bp = torch.tanh(self.B)
        C = torch.tanh(self.C)
        D = torch.tanh(self.D)

        v_state = v[..., : self.state_dim]
        v_rest  = v[..., self.state_dim:]

        for t in range(T):
            h = A * h + Bp * v_state[:, t, :]
            y_t_state = C * h + D * v_state[:, t, :]
            y_t = torch.cat([y_t_state, v_rest[:, t, :]], dim=-1)
            y.append(y_t)

        y = torch.stack(y, dim=1)                     # (B, T, C)

        # 4) gated aggregation + residual
        out = self.out_proj(gate * y)
        out = self.dropout(out)
        return self.norm(residual + out)


# -----------------------------------------------------------
# ② Mobile-Mamba backbone  (multi-channel ready)
# -----------------------------------------------------------
class MobileMambaSCA(nn.Module):
    """
    Side-Channel backbone:
      • multi-scale convolution stem
      • stacked Real-Mamba blocks
      • global pooling + classifier
    """

    def __init__(self,
                 input_len: int = 10000,
                 in_ch: int = 1,          # ★ 支持多通道
                 dim: int = 128,
                 num_blocks: int = 6,
                 num_classes: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # multi-scale convolution stem
        self.embed_33 = nn.Conv1d(in_ch, dim // 2, kernel_size=33,
                                  padding=16, stride=2, bias=False)
        self.embed_51 = nn.Conv1d(in_ch, dim // 2, kernel_size=51,
                                  padding=25, stride=2, bias=False)

        # Real-Mamba blocks
        self.blocks = nn.Sequential(*[
            RealMambaBlock(dim, state_dim=dim // 8,
                           conv_kernel=7, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状支持两种：
          • (B, T)         → 自动 unsqueeze 为 (B, 1, T)
          • (B, C, T)      → C = in_ch
        """
        if x.ndim == 2:                     # (B, T)
            x = x.unsqueeze(1)              # → (B, 1, T)

        # multi-scale conv fusion
        x1 = self.embed_33(x)               # (B, dim/2, T/2)
        x2 = self.embed_51(x)               # (B, dim/2, T/2)
        x  = torch.cat([x1, x2], dim=1)     # (B, dim,  T/2)

        x = x.permute(0, 2, 1)              # (B, T', C)
        x = self.blocks(x)                  # Real-Mamba stack
        x = x.permute(0, 2, 1)              # (B, C, T')

        x = self.pool(x).squeeze(-1)        # (B, C)
        return self.head(x)
