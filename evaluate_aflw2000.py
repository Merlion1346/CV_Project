"""
AFLW2000-3D Evaluation — HopeNet baseline protocol
Reference: https://github.com/natanielruiz/deep-head-pose/blob/master/code/test_hopenet.py

Metric : per-axis MAE (degrees) for yaw / pitch / roll
GT     : Pose_Para[:, :3] from each .mat file  (radians → degrees)
Pred   : EfficientNetHeadPose output × 90°  (denormalized)

Usage:
    python evaluate_aflw2000.py --checkpoint checkpoints/best.pth \\
                                --data_dir   AFLW2000/
"""

import os
import argparse

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

from model import EfficientNetHeadPose, HeadPoseLoss


# ── Constants ─────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
ANGLE_MAX     = 99.0   # must match training normalization (÷99°)


# ── Dataset ───────────────────────────────────
class AFLW2000Dataset(Dataset):
    """
    Pairs every .jpg with its .mat file in data_dir.
    Returns (image_tensor, angles_degrees) where
    angles_degrees = [yaw, pitch, roll] in degrees.

    Face is cropped from pt2d landmarks (same strategy as HopeNet),
    then resized to img_size × img_size.
    """

    def __init__(self, data_dir: str, img_size: int = 224):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Collect paired (jpg, mat) file stems
        stems = set()
        for fname in os.listdir(data_dir):
            if fname.lower().endswith('.jpg'):
                stem = os.path.splitext(fname)[0]
                mat_path = os.path.join(data_dir, stem + '.mat')
                if os.path.exists(mat_path):
                    stems.add(stem)

        self.samples = sorted(stems)
        print(f"[AFLW2000] Found {len(self.samples)} paired samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem     = self.samples[idx]
        img_path = os.path.join(self.data_dir, stem + '.jpg')
        mat_path = os.path.join(self.data_dir, stem + '.mat')

        # Ground-truth angles — same as HopeNet datasets.AFLW2000
        mat   = scipy.io.loadmat(mat_path)
        pose  = mat['Pose_Para'][0, :3]          # [pitch, yaw, roll] in radians
        pitch = float(pose[0]) * 180.0 / np.pi
        yaw   = float(pose[1]) * 180.0 / np.pi
        roll  = float(pose[2]) * 180.0 / np.pi
        angles = torch.tensor([yaw, pitch, roll], dtype=torch.float32)

        # Face crop from pt2d landmarks  (HopeNet strategy)
        img  = Image.open(img_path).convert('RGB')
        w, h = img.size
        pt2d = mat['pt2d']                        # (2, N)
        x_min = float(pt2d[0].min())
        y_min = float(pt2d[1].min())
        x_max = float(pt2d[0].max())
        y_max = float(pt2d[1].max())

        # Expand bounding box by 20 % on each side (same as HopeNet)
        k = abs(x_max - x_min) * 0.2
        x_min = max(0,     x_min - 2 * k)
        x_max = min(w - 1, x_max + 2 * k)
        y_min = max(0,     y_min - 2 * k)
        y_max = min(h - 1, y_max + 2 * k)

        img = img.crop((x_min, y_min, x_max, y_max))
        img = self.transform(img)

        return img, angles


# ── Evaluation loop ───────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: HeadPoseLoss):
    model.eval()

    yaw_err   = 0.0
    pitch_err = 0.0
    roll_err  = 0.0
    total     = 0

    iterable = tqdm(loader, desc="Evaluating") if USE_TQDM else loader

    for images, angles in iterable:
        images = images.to(device, non_blocking=True)
        bs     = angles.size(0)
        total += bs

        logits   = model(images)                         # (B, 3, N_BINS) bin logits
        pred_deg = criterion.predict(logits).cpu()       # soft-argmax → degrees

        label_yaw   = angles[:, 0]
        label_pitch = angles[:, 1]
        label_roll  = angles[:, 2]

        yaw_err   += torch.sum(torch.abs(pred_deg[:, 0] - label_yaw)).item()
        pitch_err += torch.sum(torch.abs(pred_deg[:, 1] - label_pitch)).item()
        roll_err  += torch.sum(torch.abs(pred_deg[:, 2] - label_roll)).item()

    mae_yaw   = yaw_err   / total
    mae_pitch = pitch_err / total
    mae_roll  = roll_err  / total
    mae_mean  = (mae_yaw + mae_pitch + mae_roll) / 3.0

    return {
        "mae_yaw":   mae_yaw,
        "mae_pitch": mae_pitch,
        "mae_roll":  mae_roll,
        "mae_mean":  mae_mean,
        "n_samples": total,
    }


# ── Print results ─────────────────────────────
def print_results(results: dict):
    n = results["n_samples"]
    print(f"\n{'='*52}")
    print(f"  AFLW2000-3D Evaluation  ({n} samples)")
    print(f"  Baseline protocol: HopeNet (MAE in degrees)")
    print(f"{'='*52}")
    print(f"  {'Axis':<10} {'MAE':>10}")
    print(f"  {'-'*22}")
    print(f"  {'Yaw':<10} {results['mae_yaw']:>9.4f}°")
    print(f"  {'Pitch':<10} {results['mae_pitch']:>9.4f}°")
    print(f"  {'Roll':<10} {results['mae_roll']:>9.4f}°")
    print(f"  {'-'*22}")
    print(f"  {'Mean':<10} {results['mae_mean']:>9.4f}°")
    print(f"{'='*52}\n")

    # HopeNet paper format for easy comparison
    print("  HopeNet-style summary:")
    print(f"  Yaw: {results['mae_yaw']:.4f}, "
          f"Pitch: {results['mae_pitch']:.4f}, "
          f"Roll: {results['mae_roll']:.4f}")


# ── CLI ───────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate on AFLW2000-3D using the HopeNet protocol."
    )
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--data_dir",    type=str, default="data/AFLW2000",
                   help="Directory containing AFLW2000 .jpg + .mat files")
    p.add_argument("--variant",     type=str, default="b0",
                   choices=["b0","b1","b2","b3","b4","b5","b6","b7"],
                   help="EfficientNet variant (overridden by checkpoint if present)")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    ckpt    = torch.load(args.checkpoint, map_location=device)
    variant = ckpt.get("variant", args.variant)
    print(f"[Eval] Variant: {variant} | Checkpoint: {args.checkpoint}")

    model     = EfficientNetHeadPose(variant=variant, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = HeadPoseLoss().to(device)

    dataset = AFLW2000Dataset(args.data_dir, img_size=args.img_size)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    results = evaluate(model, loader, device, criterion)
    print_results(results)


if __name__ == "__main__":
    main()
