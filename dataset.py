"""
AIHub KFace Dataset (Dataset #83)
Filename format: {ID}_{accessory}_{lighting}_{expression}_{pose}.jpg
"""

import os
from typing import Optional, Dict

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Pose code → (yaw, pitch, roll) mapping
#
# Source: AIHub KFace official Camera table
#   horizontal = yaw  (+right / -left)
#   vertical   = pitch (+up   / -down)
# ─────────────────────────────────────────────
POSE_MAP = {
    # pitch 0° — horizontal rotation only
    "C1":  ( 90,   0, 0), "C2":  ( 75,   0, 0), "C3":  ( 60,   0, 0),
    "C4":  ( 45,   0, 0), "C5":  ( 30,   0, 0), "C6":  ( 15,   0, 0),
    "C7":  (  0,   0, 0), "C8":  (-15,   0, 0), "C9":  (-30,   0, 0),
    "C10": (-45,   0, 0), "C11": (-60,   0, 0), "C12": (-75,   0, 0),
    "C13": (-90,   0, 0),
    # pitch +30° — camera below looking up
    "C14": ( 15,  30, 0), "C15": (  0,  30, 0), "C16": (-15,  30, 0),
    # pitch -15° — camera above looking down
    "C17": ( 15, -15, 0), "C18": (-45, -15, 0),
    "C19": ( 45, -15, 0), "C20": (-30, -15, 0),
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def parse_filename(filepath: str) -> Optional[dict]:
    """
    Parse an AIHub KFace filename and return a record dict.
    Returns None if the pose code is not in POSE_MAP.
    """
    stem  = os.path.splitext(os.path.basename(filepath))[0]
    parts = stem.split("_")
    if len(parts) < 5:
        return None

    code = parts[4].strip().upper()
    if code.isdigit():
        code = code.zfill(2)

    angles = POSE_MAP.get(code)
    if angles is None:
        return None

    yaw, pitch, roll = angles
    return {
        "path":      filepath,
        "pose_code": code,
        "yaw":       float(yaw),
        "pitch":     float(pitch),
        "roll":      float(roll),
    }


def scan_pose_codes(
    root_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
    max_samples: int = 500,
) -> set:
    """Sample up to max_samples files and report unmapped pose codes."""
    codes, count = set(), 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) >= 5:
                    codes.add(parts[4])
                count += 1
                if count >= max_samples:
                    break
        if count >= max_samples:
            break

    print(f"[Scan] Pose codes found ({count} files sampled): {sorted(codes)}")
    unmapped = codes - set(POSE_MAP.keys())
    if unmapped:
        print(f"[Scan] WARNING — unmapped codes: {sorted(unmapped)}")
    return codes


def build_dataframe(
    root_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    """Walk root_dir, parse every image filename, return a DataFrame."""
    scan_pose_codes(root_dir, extensions)

    records, skipped = [], 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                rec = parse_filename(os.path.join(dirpath, fname))
                if rec:
                    records.append(rec)
                else:
                    skipped += 1

    df = pd.DataFrame(records)
    print(f"[Dataset] Parsed: {len(df)} | Skipped: {skipped}")
    if len(df):
        print(df[["yaw", "pitch", "roll"]].describe().to_string())
    return df


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
def get_transforms(mode: str = "train", img_size: int = 224) -> transforms.Compose:
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
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


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class HeadPoseDataset(Dataset):
    """
    Returns (image, angles: Tensor[3]) — [yaw, pitch, roll].
    task 인자는 하위 호환성을 위해 유지하지만 무시됩니다.
    """

    def __init__(self, df: pd.DataFrame, transform=None, task: str = "regression"):
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


# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────
def get_dataloaders(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    batch_size: int = 32,
    img_size: int = 224,
    task: str = "regression",   # 하위 호환성 유지, 무시됨
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Split df into train/val/test and return a dict of DataLoaders."""
    train_df, temp_df = train_test_split(
        df, test_size=val_ratio + test_ratio, random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed,
    )
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    pin    = torch.cuda.is_available()
    splits = {
        "train": (train_df, "train"),
        "val":   (val_df,   "val"),
        "test":  (test_df,  "val"),
    }
    return {
        name: DataLoader(
            HeadPoseDataset(sdf, get_transforms(mode, img_size)),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        for name, (sdf, mode) in splits.items()
    }
