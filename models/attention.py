"""
attention.py  –  Unified Multi-Scale Hybrid Attention
======================================================
Combines the two existing attention styles:
  • MultiHeadSpatialAttention  (SelfBlockAttention.py)   – same-kernel Q/K/V
  • Somthing_MultiKernel       (CrissCrossAttention.py)  – criss-cross different kernels

New module: MultiScaleHybridAttention
  • Runs both branches in parallel with learnable fusion weights
  • Optional channel-squeeze-excitation gate per head
  • Drop-in replacement: same signature as the originals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# 1.  Original SelfBlock branch  (from SelfBlockAttention.py)
# ─────────────────────────────────────────────────────────────────
class SelfBlockBranch(nn.Module):
    """Symmetric-kernel multi-head spatial attention (original design)."""

    def __init__(self, in_channels, head_dim=64, num_heads=8,
                 kernels=None, img_size=256):
        super().__init__()
        if kernels is None:
            kernels = [3, 3, 3, 5, 5, 5, 7, 7]
        if len(kernels) < num_heads:
            kernels = kernels + [3] * (num_heads - len(kernels))
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.img_size  = img_size

        self.q_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in kernels])
        self.k_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in kernels])
        self.v_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in kernels])

    @staticmethod
    def _pad(img_size, k):
        return ((img_size - 1) - img_size + k) // 2

    @staticmethod
    def _softmax(x):
        B, C, H, W = x.shape
        return F.softmax(x.view(B, C, -1), dim=2).view(B, C, H, W)

    def forward(self, x):
        outs = []
        for i in range(self.num_heads):
            q = self.q_convs[i](x)
            k = self.k_convs[i](x)
            v = self.v_convs[i](x)
            _, _, H, W = q.shape
            w = self._softmax(q * k / math.sqrt(self.head_dim))
            outs.append(w * v)
        return torch.cat(outs, dim=1)          # [B, num_heads*head_dim, H, W]


# ─────────────────────────────────────────────────────────────────
# 2.  CrissCross branch  (from CrissCrossAttention.py)
# ─────────────────────────────────────────────────────────────────
class CrissCrossBranch(nn.Module):
    """Asymmetric / criss-cross kernel multi-head spatial attention."""

    # Default preset matching 'preset' from original code
    _PRESET_Q = [3, 3, 3, 5, 5, 5, 7,  7]
    _PRESET_K = [5, 5, 3, 7, 5, 3, 7, 11]
    _PRESET_V = [3, 3, 3, 5, 5, 5, 7,  7]

    def __init__(self, in_channels, head_dim=64, num_heads=8,
                 q_kernels=None, k_kernels=None, v_kernels=None,
                 img_size=256):
        super().__init__()
        q_k = (q_kernels or self._PRESET_Q)[:num_heads]
        k_k = (k_kernels or self._PRESET_K)[:num_heads]
        v_k = (v_kernels or self._PRESET_V)[:num_heads]

        self.num_heads = num_heads
        self.head_dim  = head_dim

        self.q_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in q_k])
        self.k_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in k_k])
        self.v_convs = nn.ModuleList([
            nn.Conv2d(in_channels, head_dim, k, 1,
                      self._pad(img_size, k)) for k in v_k])

    @staticmethod
    def _pad(h, k):
        return ((h - 1) - h + k) // 2

    @staticmethod
    def _softmax(x):
        B, C, H, W = x.shape
        return F.softmax(x.view(B, C, -1), dim=2).view(B, C, H, W)

    def forward(self, x):
        outs = []
        for i in range(self.num_heads):
            q = self.q_convs[i](x)
            k = self.k_convs[i](x)
            v = self.v_convs[i](x)
            _, _, H, W = q.shape
            w = self._softmax(q * k / math.sqrt(self.head_dim))
            outs.append(w * v)
        return torch.cat(outs, dim=1)          # [B, num_heads*head_dim, H, W]


# ─────────────────────────────────────────────────────────────────
# 3.  Channel Squeeze-and-Excitation gate (lightweight)
# ─────────────────────────────────────────────────────────────────
class SEGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * s


# ─────────────────────────────────────────────────────────────────
# 4.  NEW  –  MultiScaleHybridAttention
# ─────────────────────────────────────────────────────────────────
class MultiScaleHybridAttention(nn.Module):
    """
    Fuses SelfBlock + CrissCross branches with learnable alpha weights
    and optional SE-gating.

    Args:
        in_size      : spatial size of input feature map (H == W assumed)
        in_channels  : input channel count
        out_channels : output channel count  (default = in_channels)
        num_heads    : attention heads per branch  (default 8)
        head_dim     : channels per head           (default 64)
        use_se       : add SE-gate on fused features
    """

    def __init__(self, in_size, in_channels, out_channels=None,
                 num_heads=8, head_dim=64,
                 use_se=True,
                 sb_kernels=None,          # SelfBlock kernels
                 cc_q_kernels=None,        # CrissCross Q-kernels
                 cc_k_kernels=None,
                 cc_v_kernels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels  = in_channels
        self.out_channels = out_channels
        feat_ch = num_heads * head_dim      # output of each branch

        # ── two branches ──────────────────────────────────────────
        self.sb_branch = SelfBlockBranch(
            in_channels, head_dim, num_heads,
            kernels=sb_kernels, img_size=in_size)

        self.cc_branch = CrissCrossBranch(
            in_channels, head_dim, num_heads,
            q_kernels=cc_q_kernels,
            k_kernels=cc_k_kernels,
            v_kernels=cc_v_kernels,
            img_size=in_size)

        # ── learnable per-branch fusion scalar ────────────────────
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5]))

        # ── optional SE gate on concatenated features ─────────────
        self.se = SEGate(feat_ch * 2) if use_se else nn.Identity()

        # ── final 1×1 projection ──────────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        w = F.softmax(self.alpha, dim=0)          # normalise weights
        sb = self.sb_branch(x) * w[0]
        cc = self.cc_branch(x) * w[1]
        fused = self.se(torch.cat([sb, cc], dim=1))
        return self.proj(fused)


# ─────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(2, 64, 64, 64)
    m = MultiScaleHybridAttention(in_size=64, in_channels=64,
                                   out_channels=64, num_heads=8, head_dim=32)
    print("Output:", m(x).shape)           # [2, 64, 64, 64]
    print("alpha weights:", F.softmax(m.alpha, dim=0).detach())
