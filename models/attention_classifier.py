"""
attention_classifier.py  –  HybridAttentionClassifier
=======================================================
Replaces the plain ConvNeXt backbone with:
  ConvNeXt-Tiny  (pretrained)
    └─ insert MultiScaleHybridAttention after stage 2 and stage 4
    └─ global average pool
    └─ 2-layer MLP head with dropout

Also exports:
  • ResNetHybridClassifier  – lighter ResNet-50 backbone variant
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, resnet50, ResNet50_Weights
from models.attention import MultiScaleHybridAttention


# ──────────────────────────────────────────────────────────────────
# Helper – wrap a backbone stage with an attention enrichment step
# ──────────────────────────────────────────────────────────────────
class AttentionEnrichedStage(nn.Module):
    def __init__(self, stage, out_ch, img_size, num_heads=4, head_dim=32):
        super().__init__()
        self.stage = stage
        self.attn  = MultiScaleHybridAttention(
            in_size=img_size,
            in_channels=out_ch,
            out_channels=out_ch,
            num_heads=num_heads,
            head_dim=head_dim,
            use_se=True,
        )

    def forward(self, x):
        x = self.stage(x)
        return self.attn(x)


# ──────────────────────────────────────────────────────────────────
# 1.  HybridAttentionClassifier  (ConvNeXt backbone)
# ──────────────────────────────────────────────────────────────────
class HybridAttentionClassifier(nn.Module):
    """
    ConvNeXt-Tiny with MultiScaleHybridAttention injected at two
    intermediate feature levels.

    Args:
        num_classes    : number of output classes
        input_channels : 3 = RGB only, 6 = RGB + masked-RGB (your format)
        img_size       : spatial input size (assumed square, default 256)
        dropout        : dropout in MLP head
    """

    def __init__(self, num_classes=4, input_channels=6,
                 img_size=256, dropout=0.4,
                 num_heads=4, head_dim=32):
        super().__init__()
        backbone = convnext_tiny(weights="DEFAULT")

        # Patch first conv to accept your input_channels
        backbone.features[0][0] = nn.Conv2d(
            input_channels, 96, kernel_size=4, stride=4)

        # ConvNeXt-Tiny stage spatial sizes at 256 input:
        #   stage0 (stem)   → 64×64 ch=96
        #   stage1 (dblock) → 64×64 ch=96
        #   stage2 (down)   → 32×32 ch=192
        #   stage3 (dblock) → 32×32 ch=192
        #   stage4 (down)   → 16×16 ch=384
        #   stage5 (dblock) → 16×16 ch=384
        #   stage6 (down)   →  8×8  ch=768
        #   stage7 (dblock) →  8×8  ch=768

        # Wrap stages 3 and 5 with attention
        backbone.features[3] = AttentionEnrichedStage(
            backbone.features[3], out_ch=192,
            img_size=img_size // 8,
            num_heads=num_heads, head_dim=head_dim)

        backbone.features[5] = AttentionEnrichedStage(
            backbone.features[5], out_ch=384,
            img_size=img_size // 16,
            num_heads=num_heads, head_dim=head_dim)

        self.features = backbone.features

        # MLP head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


# ──────────────────────────────────────────────────────────────────
# 2.  ResNetHybridClassifier  (lighter, good baseline comparison)
# ──────────────────────────────────────────────────────────────────
class ResNetHybridClassifier(nn.Module):
    """
    ResNet-50 backbone with a MultiScaleHybridAttention refinement
    applied to layer3 output (1024 ch, ~16×16 at 256 input).

    Useful as a second model to compare against the ConvNeXt variant.
    """

    def __init__(self, num_classes=4, input_channels=6,
                 img_size=256, dropout=0.4,
                 num_heads=4, head_dim=32):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Adjust first conv for multi-channel input
        orig = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1     # 256 ch  64×64
        self.layer2 = backbone.layer2     # 512 ch  32×32
        self.layer3 = backbone.layer3     # 1024 ch 16×16
        self.layer4 = backbone.layer4     # 2048 ch  8×8

        self.attn = MultiScaleHybridAttention(
            in_size=img_size // 16,
            in_channels=1024,
            out_channels=1024,
            num_heads=num_heads,
            head_dim=head_dim,
            use_se=True,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attn(self.layer3(x))
        x = self.layer4(x)
        return self.head(x)


# ──────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(2, 6, 256, 256)

    m1 = HybridAttentionClassifier(num_classes=4, input_channels=6)
    print("HybridConvNeXt:", m1(x).shape)

    m2 = ResNetHybridClassifier(num_classes=4, input_channels=6)
    print("HybridResNet50:", m2(x).shape)
