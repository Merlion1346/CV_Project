"""
EfficientNet Head Pose — Training Script (Regression Only, 300W-LP)
Usage:
    python train.py --data_dir /path/to/300W_LP --batch_size 64 --variant b0
"""

import os
import math
import random
import argparse
import time
from glob import glob
from typing import Optional

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
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TB = True
except ImportError:
    USE_TB = False


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


COLAB_OUTPUT_DIR = "/content/drive/MyDrive/cv_checkpoint"

from model import EfficientNetHeadPose, HeadPoseLoss


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_ANGLE_MAX        = torch.tensor([99.0, 99.0, 99.0])
_ANGLE_MAX_DEVICE: dict = {}


# ─────────────────────────────────────────────
# Angle helpers
# ─────────────────────────────────────────────
def normalize(angles, device):
    if device not in _ANGLE_MAX_DEVICE:
        _ANGLE_MAX_DEVICE[device] = _ANGLE_MAX.to(device)
    return angles / _ANGLE_MAX_DEVICE[device]

def to_degrees(norm):
    return norm.cpu() * _ANGLE_MAX


# ─────────────────────────────────────────────
# 300W-LP data loading
# ─────────────────────────────────────────────
def parse_mat(mat_path: str) -> Optional[tuple]:
    """300W-LP .mat → (yaw, pitch, roll) in degrees."""
    try:
        mat  = sio.loadmat(mat_path)
        pose = mat["Pose_Para"][0]
        yaw   = math.degrees(pose[1])
        pitch = math.degrees(pose[0])
        roll  = math.degrees(pose[2])
        yaw   = max(-99.0, min(99.0, yaw))
        pitch = max(-99.0, min(99.0, pitch))
        roll  = max(-99.0, min(99.0, roll))
        return yaw, pitch, roll
    except Exception:
        return None


def build_dataframe(root_dir: str) -> pd.DataFrame:
    records, skipped = [], 0
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
        records.append({"path": img_path, "yaw": yaw, "pitch": pitch, "roll": roll})

    df = pd.DataFrame(records)
    print(f"[300W-LP] Parsed: {len(df)} | Skipped: {skipped}")
    return df


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


class HeadPoseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, flip_prob: float = 0.5):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        yaw, pitch, roll = row["yaw"], row["pitch"], row["roll"]
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            yaw = -yaw
        if self.transform:
            img = self.transform(img)
        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return img, angles


# ─────────────────────────────────────────────
# Metrics / utils
# ─────────────────────────────────────────────
def mae_per_axis(pred_norm, tgt_norm):
    return (to_degrees(pred_norm) - to_degrees(tgt_norm)).abs().mean(dim=0)

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
    total_loss, all_mae, skipped = 0.0, [], 0

    bar = tqdm(loader, desc="  train", leave=False) if USE_TQDM else loader

    for i, (imgs, angles) in enumerate(bar):
        imgs    = imgs.to(device, non_blocking=True)
        targets = normalize(angles.to(device, non_blocking=True), device)

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
        print(f"  [WARN] Skipped {skipped} batches (NaN)")

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
    total_loss, all_mae = 0.0, []

    for imgs, angles in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = normalize(angles.to(device, non_blocking=True), device)

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
# Log / Plot
# ─────────────────────────────────────────────
def _style_ax(ax, title: str, ylabel: str):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_training_curves(history: list[dict], output_dir: str):
    epochs, train_loss, val_loss = [], [], []
    mae_ep, mae_yaw, mae_pitch, mae_roll = [], [], [], []
    for r in history:
        epochs.append(r["epoch"])
        train_loss.append(r["train_loss"])
        val_loss.append(r["val_loss"])
        if r["val_mae_yaw"] is not None:
            mae_ep.append(r["epoch"])
            mae_yaw.append(r["val_mae_yaw"])
            mae_pitch.append(r["val_mae_pitch"])
            mae_roll.append(r["val_mae_roll"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss,   label="Val Loss")
    _style_ax(axes[0], "Training & Validation Loss", "Loss")

    if mae_ep:
        axes[1].plot(mae_ep, mae_yaw,   label="Yaw MAE")
        axes[1].plot(mae_ep, mae_pitch, label="Pitch MAE")
        axes[1].plot(mae_ep, mae_roll,  label="Roll MAE")
        _style_ax(axes[1], "Validation MAE per Axis", "MAE (degrees)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plot.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    if _is_colab():
        if not os.path.isdir("/content/drive/MyDrive"):
            raise RuntimeError(
                "[Colab] Google Drive not mounted.\n"
                "  Run first:  from google.colab import drive; drive.mount('/content/drive')"
            )
        print(f"[Colab] Saving to Google Drive → {args.output_dir}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[Train] Device: {device} | AMP: {use_amp}")

    if use_amp:
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Data ──────────────────────────────────
    df = build_dataframe(args.data_dir)
    if len(df) == 0:
        print("[ERROR] No data found. Check --data_dir path.")
        return

    train_df, temp_df = train_test_split(df, test_size=args.val_ratio + 0.05, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.05 / (args.val_ratio + 0.05), random_state=42)
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    pin = torch.cuda.is_available()
    nw  = args.num_workers if args.num_workers >= 0 else min(os.cpu_count(), 8)
    loaders = {
        split: DataLoader(
            HeadPoseDataset(sdf, get_transforms(mode, args.img_size)),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=(nw > 0),
        )
        for split, sdf, mode in [
            ("train", train_df, "train"),
            ("val",   val_df,   "val"),
            ("test",  test_df,  "val"),
        ]
    }

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
    start_epoch    = 1
    log_history: list[dict] = []
    best_ckpt_path = os.path.join(args.output_dir, "best.pth")
    last_ckpt_path = os.path.join(args.output_dir, "last.pth")

    # ── Resume ────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        # Restore phase-2 param group before loading optimizer state
        if ckpt["epoch"] >= args.warmup_epochs:
            model.unfreeze_top_blocks(num_blocks=3)
            backbone_lr = args.lr * 0.05
            optimizer.add_param_group({
                "params":       [p for p in model.features.parameters() if p.requires_grad],
                "lr":           backbone_lr,
                "weight_decay": args.weight_decay,
            })
            scheduler.base_lrs.append(backbone_lr)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_mae    = ckpt["best_metric"]
        start_epoch = ckpt["epoch"] + 1
        log_history = ckpt.get("log_history", [])
        print(f"[Resume] Epoch {ckpt['epoch']} loaded → continuing from epoch {start_epoch}")
    elif args.resume:
        print(f"[Resume] No checkpoint at {last_ckpt_path}, starting fresh")

    # ── Log file ──────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb")) if USE_TB else None
    if writer:
        print(f"[TensorBoard] tensorboard --logdir {os.path.join(args.output_dir, 'tb')} --port 30011")
    log_path = os.path.join(args.output_dir, "train_log.csv")
    csv_mode = "a" if args.resume and os.path.exists(log_path) else "w"
    log_f    = open(log_path, csv_mode, newline="")
    log_csv  = csv.writer(log_f)
    if csv_mode == "w":
        log_csv.writerow(["epoch", "train_loss", "val_loss",
                          "val_mae_yaw", "val_mae_pitch", "val_mae_roll"])
    print(f"[Log] {log_path}")

    # ── Epoch loop ────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if epoch == args.warmup_epochs + 1 and start_epoch <= args.warmup_epochs + 1:
            print("[Train] Phase 2 — backbone top blocks unfrozen")
            model.unfreeze_top_blocks(num_blocks=3)
            backbone_lr = args.lr * 0.05
            optimizer.add_param_group({
                "params":       [p for p in model.features.parameters() if p.requires_grad],
                "lr":           backbone_lr,
                "weight_decay": args.weight_decay,
            })
            scheduler.base_lrs.append(backbone_lr)
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr.append(backbone_lr)

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

        # ── CSV + plot ────────────────────────
        row = {
            "epoch":         epoch,
            "train_loss":    train_m["loss"],
            "val_loss":      val_m["loss"],
            "val_mae_yaw":   val_m.get("mae_yaw"),
            "val_mae_pitch": val_m.get("mae_pitch"),
            "val_mae_roll":  val_m.get("mae_roll"),
        }
        log_history.append(row)
        log_csv.writerow([v if v is not None else "" for v in row.values()])
        log_f.flush()
        plot_training_curves(log_history, args.output_dir)

        if writer:
            writer.add_scalar("Loss/train", train_m["loss"], epoch)
            writer.add_scalar("Loss/val",   val_m["loss"],   epoch)
            if "mae_yaw" in val_m:
                writer.add_scalar("MAE/yaw",   val_m["mae_yaw"],   epoch)
                writer.add_scalar("MAE/pitch", val_m["mae_pitch"], epoch)
                writer.add_scalar("MAE/roll",  val_m["mae_roll"],  epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

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

        # ── last.pth (resume용) ───────────────
        save_checkpoint({
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict(),
            "best_metric": best_mae,
            "variant":     args.variant,
            "dataset":     "300W-LP",
            "log_history": log_history,
        }, last_ckpt_path)

    log_f.close()
    if writer:
        writer.close()

    # ── Final test ────────────────────────────
    if os.path.exists(best_ckpt_path):
        print("\n[Test] Evaluating best checkpoint...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device)["model"])
        test_m = validate(model, loaders["test"], criterion, device, use_amp)
        print(f"[Test] {test_m}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="EfficientNet Head Pose — 300W-LP Training")
    p.add_argument("--data_dir",      type=str,   required=True,
                   help="Path to 300W_LP root directory")
    p.add_argument("--output_dir",    type=str,
                   default=COLAB_OUTPUT_DIR if _is_colab() else "./checkpoints")
    p.add_argument("--variant",       type=str,   default="b0",
                   choices=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=64,
                   help="VRAM 8GB 기준 B0/224 → 64, B5+ → 32 권장")
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--resume",        action="store_true",
                   help="Resume from last.pth in output_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
