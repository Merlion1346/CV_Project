"""
Head Pose Model Evaluation Script (Regression Only)
Usage:
    python evaluate.py --checkpoint ./checkpoints/best.pth --data_dir kface_data
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

from dataset import build_dataframe, HeadPoseDataset, get_transforms
from model   import EfficientNetHeadPose


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ANGLE_MAX = torch.tensor([90.0, 90.0, 90.0])

def denormalize_angles(angles_norm: torch.Tensor) -> np.ndarray:
    return (angles_norm.cpu() * ANGLE_MAX).numpy()

def angle_to_direction(yaw: float, pitch: float) -> str:
    if abs(pitch) >= 15:
        return "up" if pitch > 0 else "down"
    if abs(yaw) <= 15:
        return "front"
    return "left" if yaw < 0 else "right"


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()

    all_pred_angles = []   # (N, 3)
    all_true_angles = []

    iterable = tqdm(loader, desc="Evaluating") if USE_TQDM else loader

    for imgs, angles in iterable:
        imgs    = imgs.to(device, non_blocking=True)
        targets = angles.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)   # (B, 3) normalized

        pred_deg = denormalize_angles(preds.cpu())
        true_deg = denormalize_angles(targets.cpu())

        all_pred_angles.append(pred_deg)
        all_true_angles.append(true_deg)

    all_pred = np.concatenate(all_pred_angles, axis=0)   # (N, 3)
    all_true = np.concatenate(all_true_angles, axis=0)

    err  = all_pred - all_true
    mae  = np.abs(err).mean(axis=0)        # (3,)
    rmse = np.sqrt((err ** 2).mean(axis=0))

    # 방향 일치율 (회귀값 기반)
    pred_dirs = [angle_to_direction(r[0], r[1]) for r in all_pred]
    true_dirs = [angle_to_direction(r[0], r[1]) for r in all_true]
    dir_match = np.mean([p == t for p, t in zip(pred_dirs, true_dirs)])

    # 방향별 MAE
    from collections import defaultdict
    dir_mae = defaultdict(list)
    for pd, ta, pa in zip(true_dirs, all_true, all_pred):
        dir_mae[pd].append(np.abs(pa - ta).mean())

    return {
        "mae_yaw":   float(mae[0]),
        "mae_pitch": float(mae[1]),
        "mae_roll":  float(mae[2]),
        "mae_mean":  float(mae.mean()),
        "rmse_yaw":  float(rmse[0]),
        "rmse_pitch":float(rmse[1]),
        "rmse_roll": float(rmse[2]),
        "rmse_mean": float(rmse.mean()),
        "dir_accuracy": float(dir_match),
        "dir_mae":   {k: float(np.mean(v)) for k, v in sorted(dir_mae.items())},
        "n_samples": len(all_pred),
        "_arrays":   {"pred": all_pred, "true": all_true,
                      "pred_dirs": pred_dirs, "true_dirs": true_dirs},
    }


# ─────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────
def print_results(results):
    print("\n" + "=" * 50)
    print("  REGRESSION  (degrees)")
    print("=" * 50)
    print(f"  {'Axis':<10} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*28}")
    for axis, mae_k, rmse_k in [
        ("Yaw",   "mae_yaw",   "rmse_yaw"),
        ("Pitch", "mae_pitch", "rmse_pitch"),
        ("Roll",  "mae_roll",  "rmse_roll"),
    ]:
        print(f"  {axis:<10} {results[mae_k]:>7.2f}°  {results[rmse_k]:>7.2f}°")
    print(f"  {'-'*28}")
    print(f"  {'Mean':<10} {results['mae_mean']:>7.2f}°  {results['rmse_mean']:>7.2f}°")

    print("\n" + "=" * 50)
    print(f"  DIRECTION ACCURACY  (from regression)")
    print("=" * 50)
    print(f"  Overall: {results['dir_accuracy']*100:.2f}%")
    print(f"\n  {'Direction':<10} {'MAE':>8}")
    print(f"  {'-'*22}")
    for name, mae in results["dir_mae"].items():
        bar = "█" * int((1 - mae / 90) * 20)
        print(f"  {name:<10} {mae:>7.2f}°  {bar}")

    # 방향 confusion 간이 출력
    dirs     = sorted(set(results["_arrays"]["true_dirs"]))
    pred_d   = results["_arrays"]["pred_dirs"]
    true_d   = results["_arrays"]["true_dirs"]
    print("\n" + "=" * 50)
    print("  DIRECTION CONFUSION  (row=GT, col=Pred)")
    print("=" * 50)
    header = "".join(f"{d[:5]:>7}" for d in dirs)
    print(f"  {'':>10} {header}")
    for td in dirs:
        row = "".join(
            f"{sum(p == pd and t == td for p, t in zip(pred_d, true_d)):>7}"
            for pd in dirs
        )
        print(f"  {td:<10} {row}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[Eval] Device: {device}")

    ckpt    = torch.load(args.checkpoint, map_location=device)
    variant = ckpt.get("variant", args.variant)
    print(f"[Eval] Variant: {variant} | Checkpoint: {args.checkpoint}")

    model = EfficientNetHeadPose(variant=variant, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 학습과 동일한 split 재현
    df = build_dataframe(args.data_dir)
    _, temp_df = train_test_split(df, test_size=args.val_ratio + 0.05, random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.05 / (args.val_ratio + 0.05), random_state=42
    )

    target_df = val_df if args.split == "val" else test_df
    print(f"[Eval] Split: {args.split} | Samples: {len(target_df)}")

    dataset = HeadPoseDataset(
        target_df,
        transform=get_transforms("val", args.img_size),
        task="regression",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    results = evaluate(model, loader, device, use_amp)
    print_results(results)

    if args.save:
        save_path = os.path.join(
            os.path.dirname(args.checkpoint), f"eval_{args.split}.json"
        )
        out = {k: v for k, v in results.items() if k != "_arrays"}
        out.update({"checkpoint": args.checkpoint, "split": args.split,
                    "n_samples": results["n_samples"]})
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[Eval] Results saved: {save_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Head Pose Evaluation — Regression Only")
    p.add_argument("--checkpoint",  type=str,   required=True)
    p.add_argument("--data_dir",    type=str,   required=True)
    p.add_argument("--split",       type=str,   default="val", choices=["val", "test"])
    p.add_argument("--variant",     type=str,   default="b0",
                   choices=["b0","b1","b2","b3","b4","b5","b6","b7"])
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save",        action="store_true", help="결과를 JSON으로 저장")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
