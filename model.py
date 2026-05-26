"""
EfficientNet-based Head Pose Estimation Model
Backbone: EfficientNet B0–B7 (torchvision)
Head:     Soft-argmax binned regression — yaw / pitch / roll
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
# Spatial Attention
# ─────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx    = x.amax(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


N_BINS = 66  # -99° ~ +99° 를 3° 간격으로 분할


# ─────────────────────────────────────────────
# Binned Head
# ─────────────────────────────────────────────
class BinnedHead(nn.Module):
    """각 축(yaw/pitch/roll)에 대해 N_BINS개 bin logit 출력: (B, 3, N_BINS)."""

    def __init__(self, in_features: int, n_bins: int, dropout: float = 0.4):
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3 * n_bins),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).view(x.size(0), 3, self.n_bins)


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
        self.attn     = SpatialAttention()
        self.reg_head = BinnedHead(feat_dim, N_BINS, dropout)

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
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")

    # ── Forward ───────────────────────────────
    def forward(self, x) -> torch.Tensor:
        feat = self.features(x)     # (B, C, H, W)
        feat = self.attn(feat)      # spatial attention
        feat = self.avgpool(feat)   # (B, C, 1, 1)
        feat = feat.flatten(1)      # (B, C)
        return self.reg_head(feat)  # (B, 3, N_BINS) — bin logits per axis

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
    """Soft-argmax + Huber loss (degree space).

    logits (B, 3, N_BINS) → softmax → 기댓값(도°) → Huber vs true_deg.
    delta=3.0°: 3° 이내는 이차, 초과는 선형 패널티.
    axis_weights: [yaw, pitch, roll] 순서로 loss 가중치 적용.
    """

    def __init__(self, n_bins: int = N_BINS, angle_max: float = 99.0, delta: float = 3.0,
                 axis_weights: tuple = (1.0, 1.5, 1.5)):
        super().__init__()
        self.register_buffer("bin_centers", torch.linspace(-angle_max, angle_max, n_bins))
        self.register_buffer("axis_weights", torch.tensor(axis_weights))
        self.huber = nn.HuberLoss(reduction="none", delta=delta)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, 3, N_BINS) → (B, 3) 예측 각도(도°)."""
        return (torch.softmax(logits, dim=-1) * self.bin_centers.to(logits.device)).sum(dim=-1)

    def forward(self, logits: torch.Tensor, true_deg: torch.Tensor) -> torch.Tensor:
        loss = self.huber(self.predict(logits), true_deg)   # (B, 3)
        return (loss * self.axis_weights).mean()


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetHeadPose(variant="b0", pretrained=False)
    model.count_parameters()
    out = model(torch.randn(4, 3, 224, 224))
    print("logits:", out.shape)   # (4, 3, 66)
