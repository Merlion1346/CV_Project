"""
Experiment 2: Ablation Study
- 각 구성 요소의 기여도 분석
  Exp1: Vanilla EfficientNet-B0
  Exp2: + Flip augmentation (yaw/roll 부호 반전)
  Exp3: + Random rotation (roll 보정)
  Exp4: + Weighted loss (pitch/roll 가중치)
  Exp5: + GeM pooling
  Exp6: All combined
"""

import os
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm
from PIL import Image
import glob
import scipy.io as sio
from tqdm import tqdm


# ─────────────────────────────────────────────
# GeM Pooling
# ─────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
class HeadPoseModel(nn.Module):
    def __init__(self, use_gem: bool = False):
        super().__init__()
        # EfficientNet-B0, ImageNet pretrained
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool=""
        )
        self.use_gem = use_gem
        self.pool    = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        feat_dim     = self.backbone.num_features   # 1280

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        feat = self.backbone(x)          # (B, C, H, W)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


# ─────────────────────────────────────────────
# 손실 함수
# ─────────────────────────────────────────────
def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def weighted_mse_loss(pred, target, weights=(1.0, 1.5, 1.5)):
    """pitch/roll에 가중치를 더 부여"""
    w = torch.tensor(weights, device=pred.device)
    return ((pred - target).pow(2) * w).mean()


# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────
class HeadPoseDataset(Dataset):
    def __init__(self, data_dir, transform=None,
                 flip_aug=False, rotation_aug=False, split="train"):
        self.transform    = transform
        self.flip_aug     = flip_aug
        self.rotation_aug = rotation_aug
        self.samples      = []

        mat_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
        for mat_path in mat_files:
            img_path = mat_path.replace(".mat", ".jpg")
            if not os.path.exists(img_path):
                img_path = mat_path.replace(".mat", ".png")
            if not os.path.exists(img_path):
                continue
            try:
                mat   = sio.loadmat(mat_path)
                pose  = mat["Pose_Para"][0]
                yaw   = np.degrees(float(pose[1]))
                pitch = np.degrees(float(pose[0]))
                roll  = np.degrees(float(pose[2]))
                if abs(yaw) > 99 or abs(pitch) > 99 or abs(roll) > 99:
                    continue
                self.samples.append((img_path, yaw, pitch, roll))
            except Exception:
                continue

        n = len(self.samples)
        self.samples = (self.samples[:int(n * 0.9)] if split == "train"
                        else self.samples[int(n * 0.9):])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, yaw, pitch, roll = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # ── Flip augmentation (yaw/roll 부호 반전) ──
        if self.flip_aug and np.random.rand() < 0.5:
            img  = TF.hflip(img)
            yaw  = -yaw
            roll = -roll

        # ── Rotation augmentation (roll 보정) ──
        if self.rotation_aug:
            angle = np.random.uniform(-30, 30)
            img   = TF.rotate(img, angle)
            roll  = roll + angle   # roll 레이블 보정
            roll  = max(-90, min(90, roll))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return img, label


def get_base_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_test_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"  [Train] Epoch {epoch}/{total_epochs}", leave=False,
                ncols=90, unit="batch")
    for step, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total/(step+1):.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    pbar = tqdm(loader, desc="  [Eval ]", leave=False, ncols=90, unit="batch")
    for imgs, labels in pbar:
        preds_list.append(model(imgs.to(device)).cpu())
        labels_list.append(labels)
    preds  = torch.cat(preds_list)
    labels = torch.cat(labels_list)
    mae    = (preds - labels).abs().mean(dim=0)
    return {
        "yaw_mae":   mae[0].item(),
        "pitch_mae": mae[1].item(),
        "roll_mae":  mae[2].item(),
        "mean_mae":  mae.mean().item(),
    }


# ─────────────────────────────────────────────
# Ablation 설정 6가지
# ─────────────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "name":         "Exp1_Vanilla",
        "flip_aug":     False,
        "rotation_aug": False,
        "use_gem":      False,
        "weighted_loss":False,
        "desc":         "Vanilla EfficientNet-B0",
    },
    {
        "name":         "Exp2_FlipAug",
        "flip_aug":     True,
        "rotation_aug": False,
        "use_gem":      False,
        "weighted_loss":False,
        "desc":         "+ Flip augmentation (yaw/roll 부호 반전)",
    },
    {
        "name":         "Exp3_RotAug",
        "flip_aug":     True,
        "rotation_aug": True,
        "use_gem":      False,
        "weighted_loss":False,
        "desc":         "+ Random rotation (roll 보정)",
    },
    {
        "name":         "Exp4_WeightedLoss",
        "flip_aug":     True,
        "rotation_aug": True,
        "use_gem":      False,
        "weighted_loss":True,
        "desc":         "+ Weighted loss (pitch/roll 가중치)",
    },
    {
        "name":         "Exp5_GeM",
        "flip_aug":     True,
        "rotation_aug": True,
        "use_gem":      True,
        "weighted_loss":True,
        "desc":         "+ GeM pooling",
    },
    {
        "name":         "Exp6_AllCombined",
        "flip_aug":     True,
        "rotation_aug": True,
        "use_gem":      True,
        "weighted_loss":True,
        "desc":         "All combined (최종 모델)",
    },
]


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/300W_LP")
    parser.add_argument("--test_dir",    default="data/AFLW2000")
    parser.add_argument("--output_dir",  default="results/experiment_2_ablation")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--quick_test",  action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 5

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    test_tf  = get_test_transform(args.img_size)
    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.data_dir
    test_ds  = HeadPoseDataset(test_dir, transform=test_tf, split="test")
    print(f"Test  dataset: {len(test_ds):,} samples  ({test_dir})")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results = []
    prev_mean_mae = None

    for cfg in ABLATION_CONFIGS:
        print(f"\n{'='*65}")
        print(f"  {cfg['name']}: {cfg['desc']}")
        print(f"{'='*65}")

        # 데이터셋 (설정마다 augmentation 다름)
        train_tf = get_base_transform(args.img_size)
        train_ds = HeadPoseDataset(
            args.data_dir, transform=train_tf, split="train",
            flip_aug=cfg["flip_aug"], rotation_aug=cfg["rotation_aug"]
        )
        print(f"Train dataset: {len(train_ds):,} samples  (flip={cfg['flip_aug']}, rot={cfg['rotation_aug']})")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

        model    = HeadPoseModel(use_gem=cfg["use_gem"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if cfg["weighted_loss"]:
            loss_fn = lambda p, t: weighted_mse_loss(p, t)
        else:
            loss_fn = lambda p, t: mse_loss(p, t)

        best_mae  = float("inf")
        ckpt_path = os.path.join(args.output_dir, f"{cfg['name']}_best.pth")

        for epoch in range(1, args.epochs + 1):
            loss    = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                      epoch, args.epochs)
            metrics = evaluate(model, test_loader, device)
            scheduler.step()

            is_best = metrics["mean_mae"] < best_mae
            if is_best:
                best_mae = metrics["mean_mae"]
                torch.save(model.state_dict(), ckpt_path)

            best_mark = " ★" if is_best else ""
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={loss:.4f} | "
                  f"Yaw={metrics['yaw_mae']:.2f}° "
                  f"Pitch={metrics['pitch_mae']:.2f}° "
                  f"Roll={metrics['roll_mae']:.2f}° "
                  f"Mean={metrics['mean_mae']:.2f}°"
                  f"{best_mark}")

        # 최종 평가
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final = evaluate(model, test_loader, device)

        delta = ""
        if prev_mean_mae is not None:
            d = final["mean_mae"] - prev_mean_mae
            delta = f"  (Δ {d:+.2f}°)"
        prev_mean_mae = final["mean_mae"]

        result = {
            "exp":          cfg["name"],
            "desc":         cfg["desc"],
            "flip_aug":     cfg["flip_aug"],
            "rotation_aug": cfg["rotation_aug"],
            "use_gem":      cfg["use_gem"],
            "weighted_loss":cfg["weighted_loss"],
            "yaw_mae":      round(final["yaw_mae"],   4),
            "pitch_mae":    round(final["pitch_mae"],  4),
            "roll_mae":     round(final["roll_mae"],   4),
            "mean_mae":     round(final["mean_mae"],   4),
        }
        results.append(result)
        print(f"\n  ✅ Mean MAE={final['mean_mae']:.2f}°{delta}")

    # 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*65)
    print("  ABLATION STUDY RESULTS")
    print("="*65)
    print(df[["exp", "desc", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"]].to_string(index=False))

    # 시각화
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(df))
        ax.bar(x, df["mean_mae"], color="steelblue", alpha=0.8)
        ax.axhline(y=5.0, color="red", linestyle="--", label="Target (5°)")
        ax.set_xticks(x)
        ax.set_xticklabels(df["exp"], rotation=15, ha="right")
        ax.set_ylabel("Mean MAE (°)")
        ax.set_title("Ablation Study: Each Component Contribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "ablation_plot.png"), dpi=150)
        print(f"\n  Plot saved.")
    except Exception as e:
        print(f"  Plot 생성 실패: {e}")

    print(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
