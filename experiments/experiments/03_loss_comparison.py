"""
Experiment 3: Loss Function Comparison
- MSE only
- Cross Entropy only (bin classification)
- Combined CE + MSE  ← HopeNet 방식
- Combined + Wrapped loss  ← WHENet 방식
- Focal MSE variant
"""

import os
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
# 각도 → bin 변환 (CE loss용)
# ─────────────────────────────────────────────
NUM_BINS  = 66
IDX_START = -99


def angle_to_bin(angle_deg, num_bins=NUM_BINS, start=IDX_START):
    """angle → bin index (0 ~ num_bins-1)"""
    bin_size = abs(start) * 2 / num_bins
    idx = (angle_deg - start) / bin_size
    return int(np.clip(idx, 0, num_bins - 1))


def bin_to_angle(bins_logits, num_bins=NUM_BINS, start=IDX_START):
    """softmax 기댓값으로 연속 각도 복원"""
    bin_size  = abs(start) * 2 / num_bins
    centers   = torch.arange(num_bins, dtype=torch.float32,
                              device=bins_logits.device) * bin_size + start + bin_size / 2
    probs     = F.softmax(bins_logits, dim=-1)
    return (probs * centers).sum(dim=-1)


# ─────────────────────────────────────────────
# 손실 함수 5가지
# ─────────────────────────────────────────────
def loss_mse(pred_reg, target_reg, pred_cls=None, target_cls=None):
    return F.mse_loss(pred_reg, target_reg)


def loss_ce(pred_reg, target_reg, pred_cls, target_cls):
    """Cross Entropy (분류 헤드 필요)"""
    ce = sum(F.cross_entropy(pred_cls[:, i, :], target_cls[:, i])
             for i in range(3)) / 3
    return ce


def loss_combined(pred_reg, target_reg, pred_cls, target_cls, alpha=1.0):
    """CE + alpha * MSE  (HopeNet)"""
    ce  = sum(F.cross_entropy(pred_cls[:, i, :], target_cls[:, i])
              for i in range(3)) / 3
    mse = F.mse_loss(pred_reg, target_reg)
    return ce + alpha * mse


def loss_wrapped(pred_reg, target_reg, pred_cls, target_cls, alpha=1.0):
    """Wrapped loss + CE  (WHENet)"""
    ce   = sum(F.cross_entropy(pred_cls[:, i, :], target_cls[:, i])
               for i in range(3)) / 3
    diff = (pred_reg - target_reg).abs()
    wrapped = torch.min(diff, 360 - diff).mean()
    return ce + alpha * wrapped


def loss_focal_mse(pred_reg, target_reg, pred_cls=None, target_cls=None, gamma=2.0):
    """Focal-weighted MSE: 큰 오차에 더 집중"""
    err    = (pred_reg - target_reg).abs()
    weight = err.detach().pow(gamma)
    return (weight * err.pow(2)).mean()


LOSS_CONFIGS = [
    {"name": "MSE_only",          "fn": loss_mse,       "need_cls": False},
    {"name": "CE_only",           "fn": loss_ce,        "need_cls": True},
    {"name": "Combined_CE_MSE",   "fn": loss_combined,  "need_cls": True},
    {"name": "Combined_Wrapped",  "fn": loss_wrapped,   "need_cls": True},
    {"name": "Focal_MSE",         "fn": loss_focal_mse, "need_cls": False},
]


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
class HeadPoseModel(nn.Module):
    def __init__(self, with_cls_head: bool = False):
        super().__init__()
        self.backbone    = timm.create_model("efficientnet_b0", pretrained=True,
                                             num_classes=0)
        feat_dim         = self.backbone.num_features
        self.with_cls    = with_cls_head

        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 3),
        )
        if with_cls_head:
            # 3축 × NUM_BINS
            self.cls_head = nn.Linear(feat_dim, 3 * NUM_BINS)

    def forward(self, x):
        feat = self.backbone(x)
        reg  = self.reg_head(feat)
        if self.with_cls:
            cls = self.cls_head(feat).view(-1, 3, NUM_BINS)
            return reg, cls
        return reg, None


# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────
class HeadPoseDataset(Dataset):
    def __init__(self, data_dir, transform=None, with_bins=False, split="train"):
        self.transform  = transform
        self.with_bins  = with_bins
        self.samples    = []

        for mat_path in glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True):
            img_path = mat_path.replace(".mat", ".jpg")
            if not os.path.exists(img_path):
                img_path = mat_path.replace(".mat", ".png")
            if not os.path.exists(img_path):
                continue
            try:
                import scipy.io as sio
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
        # flip augmentation
        if np.random.rand() < 0.5:
            img   = TF.hflip(img)
            yaw   = -yaw
            roll  = -roll
        if self.transform:
            img = self.transform(img)
        label_reg = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        if self.with_bins:
            label_cls = torch.tensor(
                [angle_to_bin(yaw), angle_to_bin(pitch), angle_to_bin(roll)],
                dtype=torch.long
            )
            return img, label_reg, label_cls
        return img, label_reg, torch.zeros(3, dtype=torch.long)


def get_transform(img_size=224, augment=True):
    tf = [transforms.Resize((img_size, img_size))]
    if augment:
        tf += [transforms.ColorJitter(0.2, 0.2, 0.2)]
    tf += [transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(tf)


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, need_cls, device, epoch, total_epochs):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"  [Train] Epoch {epoch}/{total_epochs}", leave=False,
                ncols=90, unit="batch")
    for step, (imgs, labels_reg, labels_cls) in enumerate(pbar):
        imgs       = imgs.to(device)
        labels_reg = labels_reg.to(device)
        labels_cls = labels_cls.to(device)
        optimizer.zero_grad()
        pred_reg, pred_cls = model(imgs)
        if need_cls:
            loss = loss_fn(pred_reg, labels_reg, pred_cls, labels_cls)
        else:
            loss = loss_fn(pred_reg, labels_reg)
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
    for imgs, labels_reg, _ in pbar:
        pred_reg, _ = model(imgs.to(device))
        preds_list.append(pred_reg.cpu())
        labels_list.append(labels_reg)
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
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/300W_LP")
    parser.add_argument("--test_dir",    default="data/AFLW2000")
    parser.add_argument("--output_dir",  default="results/experiment_3_loss")
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

    test_tf  = get_transform(args.img_size, augment=False)
    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.data_dir
    test_ds  = HeadPoseDataset(test_dir, transform=test_tf, with_bins=False, split="test")
    print(f"Test  dataset: {len(test_ds):,} samples  ({test_dir})")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results = []
    prev_mean_mae = None

    for cfg in LOSS_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Loss: {cfg['name']}")
        print(f"{'='*60}")

        need_cls = cfg["need_cls"]
        train_tf = get_transform(args.img_size, augment=True)

        train_ds = HeadPoseDataset(args.data_dir, transform=train_tf,
                                   with_bins=need_cls, split="train")
        print(f"Train dataset: {len(train_ds):,} samples  (cls_head={need_cls})")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers, pin_memory=True)

        model     = HeadPoseModel(with_cls_head=need_cls).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_mae  = float("inf")
        ckpt_path = os.path.join(args.output_dir, f"{cfg['name']}_best.pth")

        for epoch in range(1, args.epochs + 1):
            loss    = train_one_epoch(model, train_loader, optimizer,
                                      cfg["fn"], need_cls, device,
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

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final = evaluate(model, test_loader, device)

        delta = ""
        if prev_mean_mae is not None:
            d = final["mean_mae"] - prev_mean_mae
            delta = f"  (Δ {d:+.2f}°)"
        prev_mean_mae = final["mean_mae"]

        results.append({
            "loss_fn":   cfg["name"],
            "yaw_mae":   round(final["yaw_mae"],   4),
            "pitch_mae": round(final["pitch_mae"],  4),
            "roll_mae":  round(final["roll_mae"],   4),
            "mean_mae":  round(final["mean_mae"],   4),
        })
        print(f"\n  ✅ {cfg['name']} | Mean MAE={final['mean_mae']:.2f}°{delta}")

    df = pd.DataFrame(results).sort_values("mean_mae")
    csv_path = os.path.join(args.output_dir, "loss_comparison.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("  LOSS FUNCTION COMPARISON RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")


if __name__ == "__main__":
    main()
