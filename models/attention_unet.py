"""
attention_unet.py  –  AttentionHybridUNet
==========================================
Drops MultiScaleHybridAttention into three roles inside a UNet:
  1. Bottleneck  – refines the deepest feature map
  2. Skip gates  – replaces classic Attention_block at each decoder level
  3. Encoder blocks can optionally use attention-enhanced convolutions

Architecture sketch (256×256 input):
  Enc1 64  →  Enc2 128  →  Enc3 256  →  Enc4 512  →  Bottleneck 1024
                                                              ↓
                                              Hybrid-Att bottleneck
                                                              ↓
  Dec4 512  ←  HybridSkip4  ←  Enc4
  Dec3 256  ←  HybridSkip3  ←  Enc3
  Dec2 128  ←  HybridSkip2  ←  Enc2
  Dec1  64  ←  HybridSkip1  ←  Enc1
       ↓
  Conv1×1 → output_ch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiScaleHybridAttention


# ──────────────────────────────────────────────────────────────────
# Basic building blocks
# ──────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, k, s, p, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch_in,  ch_out),
            ConvBNReLU(ch_out, ch_out),
        )
    def forward(self, x): return self.block(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(ch_in, ch_out),
        )
    def forward(self, x): return self.up(x)


# ──────────────────────────────────────────────────────────────────
# Hybrid attention skip gate
#   Replaces the classic additive Attention_block with the full
#   MultiScaleHybridAttention applied only to the skip feature map
# ──────────────────────────────────────────────────────────────────
class HybridSkipGate(nn.Module):
    """
    Gate the skip connection x with MultiScaleHybridAttention.
    g (decoder query) is used to modulate via a 1×1 conv before fusion.
    """
    def __init__(self, feat_ch, gate_ch, img_size,
                 num_heads=4, head_dim=32):
        super().__init__()
        # align gate to feat_ch
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = MultiScaleHybridAttention(
            in_size=img_size,
            in_channels=feat_ch,
            out_channels=feat_ch,
            num_heads=num_heads,
            head_dim=head_dim,
            use_se=True,
        )

    def forward(self, x, g):
        """
        x : skip feature  [B, feat_ch, H, W]
        g : decoder query [B, gate_ch, H, W]  (may differ spatially → upsample first)
        """
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:],
                              mode='bilinear', align_corners=False)
        g_proj = self.gate_proj(g)
        gated  = self.attn(x + g_proj)     # add gate hint then attend
        return gated


# ──────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────
class AttentionHybridUNet(nn.Module):
    """
    UNet with MultiScaleHybridAttention skip gates + attention bottleneck.

    Args:
        img_ch      : input channels  (3 = RGB)
        output_ch   : output channels (1 = binary mask, 4 = multi-class)
        base_ch     : base feature width  (default 64 → classic UNet schedule)
        img_size    : spatial size of the input (assumed square)
        num_heads   : heads in each attention module
        head_dim    : dimension per head
    """

    def __init__(self, img_ch=3, output_ch=1,
                 base_ch=32, img_size=256,
                 num_heads=4, head_dim=32):
        super().__init__()
        b = base_ch
        self.pool = nn.MaxPool2d(2, 2)

        # ── encoder ───────────────────────────────────────────────
        self.enc1 = DoubleConv(img_ch, b)           # 256
        self.enc2 = DoubleConv(b,      b*2)         # 128
        self.enc3 = DoubleConv(b*2,    b*4)         #  64
        self.enc4 = DoubleConv(b*4,    b*8)         #  32

        # ── bottleneck (with hybrid attention) ────────────────────
        self.bottleneck_conv = DoubleConv(b*8, b*16)  # 16
        self.bottleneck_attn = MultiScaleHybridAttention(
            in_size=img_size // 16,
            in_channels=b*16,
            out_channels=b*16,
            num_heads=num_heads,
            head_dim=head_dim,
            use_se=True,
        )

        # ── decoder + skip gates ──────────────────────────────────
        sizes = [img_size // s for s in (2, 4, 8, 16)]  # 128, 64, 32, 16

        self.up4 = UpConv(b*16, b*8)
        self.sg4 = HybridSkipGate(b*8,  b*16, sizes[3],  num_heads, head_dim)
        self.dec4 = DoubleConv(b*16, b*8)

        self.up3 = UpConv(b*8, b*4)
        self.sg3 = HybridSkipGate(b*4,  b*8,  sizes[2],  num_heads, head_dim)
        self.dec3 = DoubleConv(b*8,  b*4)

        self.up2 = UpConv(b*4, b*2)
        self.sg2 = HybridSkipGate(b*2,  b*4,  sizes[1],  num_heads, head_dim)
        self.dec2 = DoubleConv(b*4,  b*2)

        self.up1 = UpConv(b*2, b)
        self.sg1 = HybridSkipGate(b,    b*2,  sizes[0],  num_heads, head_dim)
        self.dec1 = DoubleConv(b*2,  b)

        # ── output ────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(b, output_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck_conv(self.pool(e4))
        b  = self.bottleneck_attn(b)

        # Decoder
        d4 = self.up4(b)
        s4 = self.sg4(e4, b)
        d4 = self.dec4(torch.cat([s4, d4], dim=1))

        d3 = self.up3(d4)
        s3 = self.sg3(e3, d4)
        d3 = self.dec3(torch.cat([s3, d3], dim=1))

        d2 = self.up2(d3)
        s2 = self.sg2(e2, d3)
        d2 = self.dec2(torch.cat([s2, d2], dim=1))

        d1 = self.up1(d2)
        s1 = self.sg1(e1, d2)
        d1 = self.dec1(torch.cat([s1, d1], dim=1))

        return self.out_conv(d1)


# ──────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = AttentionHybridUNet(img_ch=3, output_ch=1,
                                 img_size=256, num_heads=4, head_dim=32)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print("Output:", out.shape)   # [2, 1, 256, 256]
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
