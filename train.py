"""
EfficientNet Head Pose — Training Script (Regression Only)
Usage:
    python train.py --data_dir kface_data --batch_size 512 --variant b4 --img_size 380
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

from dataset import build_dataframe, get_dataloaders
from model   import EfficientNetHeadPose, HeadPoseLoss


# ─────────────────────────────────────────────
# Angle normalization helpers
# ─────────────────────────────────────────────
_ANGLE_MAX = torch.tensor([99.0, 99.0, 99.0])

def normalize(angles, device):
    return angles / _ANGLE_MAX.to(device)

def to_degrees(norm):
    """Denormalize → degrees (CPU)."""
    return norm.cpu() * _ANGLE_MAX


# ─────────────────────────────────────────────
# Metrics
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
    total_loss = 0.0
    all_mae    = []
    skipped    = 0

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
    total_loss = 0.0
    all_mae    = []

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
# Training loop
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
    df = build_dataframe(args.data_dir)
    loaders = get_dataloaders(
        df,
        val_ratio=args.val_ratio,
        test_ratio=0.05,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers if args.num_workers >= 0 else min(os.cpu_count(), 8),
    )

    # ── Model ─────────────────────────────────
    model = EfficientNetHeadPose(
        variant=args.variant,
        pretrained=True,
        dropout=args.dropout,
    ).to(device)

    # 300W-LP 사전학습 가중치 로드 (선택)
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Train] Loaded pretrained weights: {args.pretrained_ckpt}")

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

    best_loss      = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best.pth")

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

        train_m = train_one_epoch(model, loaders["train"], optimizer, criterion,
                                  device, use_amp, scaler)
        val_m   = validate(model, loaders["val"], criterion, device, use_amp)
        scheduler.step()

        # ── Log ───────────────────────────────
        elapsed = time.time() - t0
        log = (f"Epoch [{epoch:03d}/{args.epochs}] ({elapsed:.1f}s) | "
               f"Train: {train_m['loss']:.4f} | Val: {val_m['loss']:.4f}")
        if "mae_yaw" in val_m:
            log += (f" | Yaw:{val_m['mae_yaw']:.1f}° "
                    f"Pitch:{val_m['mae_pitch']:.1f}° "
                    f"Roll:{val_m['mae_roll']:.1f}°")
        print(log)

        # ── Save best checkpoint ───────────────
        if not np.isnan(val_m["loss"]) and val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            save_checkpoint({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_metric": best_loss,
                "variant":     args.variant,
            }, best_ckpt_path)

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
    p = argparse.ArgumentParser(description="EfficientNet Head Pose — Regression Training")
    p.add_argument("--data_dir",      type=str,   required=True)
    p.add_argument("--output_dir",    type=str,   default="./checkpoints")
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
    p.add_argument("--val_ratio",     type=float, default=0.15)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--pretrained_ckpt", type=str, default=None,
                   help="Path to 300W-LP pretrained checkpoint for finetuning")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
