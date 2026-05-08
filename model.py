"""
EfficientNet-based Head Pose Estimation Model
Backbone: EfficientNet B3–B7 (torchvision)
Head:     Regression only — yaw / pitch / roll
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
)


# ─────────────────────────────────────────────
# Channel Attention (lightweight CBAM)
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


# ─────────────────────────────────────────────
# Regression Head
# ─────────────────────────────────────────────
class RegressionHead(nn.Module):
    """Predicts normalized yaw / pitch / roll in [-1, 1]."""

    def __init__(self, in_features: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3),
        )
        # Small init on final layer → stable early loss
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────
class EfficientNetHeadPose(nn.Module):
    """
    Args:
        variant    : EfficientNet variant — "b0" | "b1" | ... | "b7"
        pretrained : use ImageNet-pretrained backbone weights
        dropout    : dropout ratio for regression head
    """

    _VARIANTS = {
        "b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        "b1": (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1, 1280),
        "b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 1408),
        "b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
        "b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
        "b5": (efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1, 2048),
        "b6": (efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1, 2304),
        "b7": (efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1, 2560),
    }

    def __init__(
        self,
        variant: str = "b0",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        assert variant in self._VARIANTS, \
            f"variant must be one of {list(self._VARIANTS)}"

        model_fn, weights, feat_dim = self._VARIANTS[variant]

        backbone      = model_fn(weights=weights if pretrained else None)
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.attn     = ChannelAttention(feat_dim)
        self.reg_head = RegressionHead(feat_dim, dropout)

        self._init_heads()

    # ── Initialization ────────────────────────
    def _init_heads(self):
        """Kaiming init for head layers only (backbone is pretrained)."""
        for head in (self.reg_head, self.attn):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────
    def forward(self, x) -> torch.Tensor:
        feat = self.features(x)     # (B, C, H, W)
        feat = self.attn(feat)      # channel attention
        feat = self.avgpool(feat)   # (B, C, 1, 1)
        feat = feat.flatten(1)      # (B, C)
        return self.reg_head(feat)  # (B, 3)  — [yaw, pitch, roll]

    # ── Backbone control ──────────────────────
    def freeze_backbone(self, freeze: bool = True):
        for p in self.features.parameters():
            p.requires_grad = not freeze
        print(f"[Model] Backbone {'frozen' if freeze else 'unfrozen'}")

    def unfreeze_top_blocks(self, num_blocks: int = 3):
        for block in list(self.features.children())[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"[Model] Top {num_blocks} backbone blocks unfrozen")

    # ── Info ──────────────────────────────────
    def count_parameters(self) -> int:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total: {total:,} | Trainable: {trainable:,}")
        return trainable


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
class HeadPoseLoss(nn.Module):
    """MSE regression loss for (yaw, pitch, roll)."""

    def __init__(self):
        super().__init__()
        self.reg_loss = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.reg_loss(preds, targets)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetHeadPose(variant="b0", pretrained=False)
    model.count_parameters()
    out = model(torch.randn(4, 3, 224, 224))
    print("angles:", out.shape)   # (4, 3)
