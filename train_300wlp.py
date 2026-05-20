"""
300W-LP Dataset for Head Pose Estimation (Regression Only)
==========================================================
Download: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

Directory structure after extraction:
    300W_LP/
    ├── AFW/
    │   ├── AFW_134212_1_0.jpg
    │   ├── AFW_134212_1_0.mat   ← pose annotation
    │   └── ...
    ├── HELEN/
    ├── IBUG/
    ├── LFPW/
    └── AFW_Flip/   ← horizontally flipped versions
    └── HELEN_Flip/
    └── ...

Usage:
    python train_300wlp.py --data_dir /path/to/300W_LP --epochs 50 --batch_size 128

    # Then finetune on KFace
    python train.py --data_dir kface_data --epochs 20 --lr 1e-4
"""

import os
import math
import argparse
import time
from glob import glob
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

from model import EfficientNetHeadPose, HeadPoseLoss


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
ANGLE_MAX_300W = torch.tensor([99.0, 99.0, 99.0])


# ─────────────────────────────────────────────
# Angle helpers
# ─────────────────────────────────────────────
def normalize_angles(angles: torch.Tensor, device) -> torch.Tensor:
    return angles / ANGLE_MAX_300W.to(device)

def denormalize_angles(norm: torch.Tensor) -> torch.Tensor:
    return norm.cpu() * ANGLE_MAX_300W


# ─────────────────────────────────────────────
# .mat file parser
# ─────────────────────────────────────────────
def parse_mat(mat_path: str) -> Optional[Tuple[float, float, float]]:
    """
    300W-LP .mat annotation → (yaw, pitch, roll) in degrees.

    Mat file contains 'Pose_Para': [pitch, yaw, roll, tx, ty, tz]
    Values are in radians. Sign convention:
        - positive yaw   = face turning right
        - positive pitch = face looking up
        - positive roll  = head tilting right
    """
    try:
        mat = sio.loadmat(mat_path)
        # Pose_Para shape: (1, 7) — [pitch, yaw, roll, tx, ty, tz, scale]
        pose = mat["Pose_Para"][0]
        pitch_rad, yaw_rad, roll_rad = pose[0], pose[1], pose[2]

        pitch = math.degrees(pitch_rad)
        yaw   = math.degrees(yaw_rad)
        roll  = math.degrees(roll_rad)

        # Clamp to valid range
        yaw   = max(-99.0, min(99.0, yaw))
        pitch = max(-99.0, min(99.0, pitch))
        roll  = max(-99.0, min(99.0, roll))

        return yaw, pitch, roll
    except Exception:
        return None


# ─────────────────────────────────────────────
# Build DataFrame
# ─────────────────────────────────────────────
def build_300wlp_dataframe(root_dir: str) -> pd.DataFrame:
    """
    Walk 300W_LP directory, parse all .mat files → DataFrame.
    Subfolders: AFW, HELEN, IBUG, LFPW (+ _Flip variants)
    """
    records = []
    skipped = 0

    img_paths = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    print(f"[300W-LP] Found {len(img_paths)} images, parsing annotations...")

    bar = tqdm(img_paths) if USE_TQDM else img_paths
    for img_path in bar:
        mat_path = os.path.splitext(img_path)[0] + ".mat"
        if not os.path.exists(mat_path):
            skipped += 1
            continue

        angles = parse_mat(mat_path)
        if angles is None:
            skipped += 1
            continue

        yaw, pitch, roll = angles
        records.append({
            "path":  img_path,
            "yaw":   yaw,
            "pitch": pitch,
            "roll":  roll,
        })

    df = pd.DataFrame(records)
    print(f"[300W-LP] Parsed: {len(df)} | Skipped: {skipped}")
    return df


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def get_transforms(mode: str = "train", img_size: int = 224) -> transforms.Compose:
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Dataset300WLP(Dataset):
    """
    300W-LP Dataset.
    Returns (image, angles: Tensor[3])  — [yaw, pitch, roll]
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        angles = torch.tensor([row["yaw"], row["pitch"], row["roll"]], dtype=torch.float32)
        return img, angles


def get_dataloaders(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    batch_size: int = 128,
    img_size: int = 224,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)}")

    pin = torch.cuda.is_available()
    return {
        "train": DataLoader(
            Dataset300WLP(train_df, get_transforms("train", img_size)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=(num_workers > 0),
        ),
        "val": DataLoader(
            Dataset300WLP(val_df, get_transforms("val", img_size)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=(num_workers > 0),
        ),
    }


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def mae_per_axis(pred_norm: torch.Tensor, tgt_norm: torch.Tensor) -> torch.Tensor:
    """MAE per axis in degrees (denormalized)."""
    return (denormalize_angles(pred_norm) - denormalize_angles(tgt_norm)).abs().mean(dim=0)

def has_nan(tensor: torch.Tensor, tag: str) -> bool:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"  [WARN] NaN/Inf — {tag}")
        return True
    return False


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    all_mae    = []
    skipped    = 0

    bar = tqdm(loader, desc="  train", leave=False) if USE_TQDM else loader

    for i, (imgs, angles) in enumerate(bar):
        imgs    = imgs.to(device, non_blocking=True)
        targets = normalize_angles(angles.to(device, non_blocking=True), device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            if has_nan(preds, f"b{i} preds"):
                skipped += 1; continue
            loss = criterion(preds, targets)

        if has_nan(loss, f"b{i} loss"):
            skipped += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if has_nan(gnorm, f"b{i} grad"):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped += 1; continue
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            all_mae.append(mae_per_axis(preds.detach(), targets).numpy())

    if skipped:
        print(f"  [WARN] Skipped {skipped} batches")

    n      = max(len(loader) - skipped, 1)
    result = {"loss": total_loss / n}
    if all_mae:
        m = np.stack(all_mae).mean(0)
        result |= {"mae_yaw": float(m[0]), "mae_pitch": float(m[1]), "mae_roll": float(m[2])}
    return result


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_mae    = []

    for imgs, angles in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = normalize_angles(angles.to(device, non_blocking=True), device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            loss  = criterion(preds, targets)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
        all_mae.append(mae_per_axis(preds, targets).numpy())

    n      = max(len(loader), 1)
    result = {"loss": total_loss / n}
    if all_mae:
        m = np.stack(all_mae).mean(0)
        result |= {"mae_yaw": float(m[0]), "mae_pitch": float(m[1]), "mae_roll": float(m[2])}
    return result


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [Saved] {path}")


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[Train] Device: {device} | AMP: {use_amp}")

    if use_amp:
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Data ──────────────────────────────────
    df = build_300wlp_dataframe(args.data_dir)
    if len(df) == 0:
        print("[ERROR] No data found. Check --data_dir path.")
        return

    loaders = get_dataloaders(
        df,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────
    model = EfficientNetHeadPose(
        variant=args.variant,
        pretrained=True,
        dropout=args.dropout,
    ).to(device)
    model.count_parameters()
    model.freeze_backbone(True)

    # ── Optimizer / Scheduler / Scaler ────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_amp,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler(device="cuda" if use_amp else "cpu", enabled=use_amp)
    criterion = HeadPoseLoss()

    best_mae       = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_300wlp.pth")

    # ── Epoch loop ────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 2: unfreeze top backbone blocks
        if epoch == args.warmup_epochs + 1:
            print("[Train] Phase 2 — backbone top blocks unfrozen")
            model.unfreeze_top_blocks(num_blocks=3)
            optimizer.add_param_group({
                "params":       [p for p in model.features.parameters() if p.requires_grad],
                "lr":           args.lr * 0.05,
                "weight_decay": args.weight_decay,
            })
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        train_m = train_one_epoch(model, loaders["train"], optimizer, criterion,
                                  device, use_amp, scaler)
        val_m   = validate(model, loaders["val"], criterion, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        log = (f"Epoch [{epoch:03d}/{args.epochs}] ({elapsed:.1f}s) | "
               f"Train: {train_m['loss']:.4f} | Val: {val_m['loss']:.4f}")
        if "mae_yaw" in val_m:
            log += (f" | Yaw:{val_m['mae_yaw']:.1f}° "
                    f"Pitch:{val_m['mae_pitch']:.1f}° "
                    f"Roll:{val_m['mae_roll']:.1f}°")
        print(log)

        val_mae_mean = (val_m.get("mae_yaw", float("inf")) +
                        val_m.get("mae_pitch", float("inf")) +
                        val_m.get("mae_roll", float("inf"))) / 3.0
        if val_mae_mean < best_mae:
            best_mae = val_mae_mean
            save_checkpoint({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_metric": best_mae,
                "variant":     args.variant,
                "dataset":     "300W-LP",
            }, best_ckpt_path)

    print(f"\n[Done] Best checkpoint: {best_ckpt_path}")
    print("Next step — finetune on KFace:")
    print(f"  python train.py --data_dir kface_data "
          f"--pretrained_ckpt {best_ckpt_path} --lr 1e-4 --epochs 30")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train on 300W-LP dataset (regression only)")
    p.add_argument("--data_dir",      type=str,   required=True,
                   help="Path to 300W_LP root directory")
    p.add_argument("--output_dir",    type=str,   default="./checkpoints")
    p.add_argument("--variant",       type=str,   default="b0",
                   choices=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
    p.add_argument("--epochs",        type=int,   default=50) #50권장
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=64,
                   help="VRAM 8GB 기준 B0/224 → 64, B5+ → 32 권장")
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
