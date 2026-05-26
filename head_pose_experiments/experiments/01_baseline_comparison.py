"""
Experiment 1: Baseline Comparison
- ResNet50    (HopeNet 기준, 25.6M)
- MobileNetV2 (경량, 3.5M)
- EfficientNet-B0 (제안 모델, 5.3M)
- EfficientNet-B1 (B0 확장, 7.8M)
모두 ImageNet pretrained 사용
"""

import os
import time
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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def log(msg: str):
    print(msg, flush=True)


# ─────────────────────────────────────────────
# 백본 정의 (4개, 모두 ImageNet pretrained)
# ─────────────────────────────────────────────
BACKBONES = {
    "resnet50":        "resnet50",
    "mobilenetv2":     "mobilenetv2_100",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
}


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
class HeadPoseModel(nn.Module):
    def __init__(self, timm_name: str):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),   # yaw, pitch, roll
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────
class HeadPoseDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        self.transform = transform
        self.samples   = []

        log(f"[Dataset] Scanning {data_dir} ({split})...")
        mat_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
        log(f"[Dataset] Found {len(mat_files)} .mat files")

        iterator = mat_files
        if tqdm is not None:
            iterator = tqdm(mat_files, desc=f"Load {split}", unit="file")

        for mat_path in iterator:
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
        log(f"[Dataset] {split}: {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, yaw, pitch, roll = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([yaw, pitch, roll], dtype=torch.float32)


def get_transform(img_size=224, augment=True):
    tf = [transforms.Resize((img_size, img_size))]
    if augment:
        tf += [transforms.RandomHorizontalFlip(0.5),
               transforms.ColorJitter(0.2, 0.2, 0.2)]
    tf += [transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])]
    return transforms.Compose(tf)


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, epoch=None):
    model.train()
    total = 0.0
    iterator = loader
    if tqdm is not None:
        desc = f"Train e{epoch}" if epoch else "Train"
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch")

    for imgs, labels in iterator:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    preds_list, labels_list = [], []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch")

    for imgs, labels in iterator:
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


def measure_fps(model, device, img_size=224, n_iter=200):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(10):
        model(dummy)
    t = time.time()
    for _ in range(n_iter):
        model(dummy)
    return round(n_iter / (time.time() - t), 1)


def measure_fps_cpu(model, img_size=224, n_iter=100):
    model_cpu = model.cpu().eval()
    dummy     = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        model_cpu(dummy)
    t = time.time()
    for _ in range(n_iter):
        model_cpu(dummy)
    return round(n_iter / (time.time() - t), 1)


def count_params(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 1)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/300W_LP")
    parser.add_argument("--test_dir",    default="data/AFLW2000")
    parser.add_argument("--output_dir",  default="results/experiment_1_baseline")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--quick_test",  action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 5

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        log(f"Device: {device}")
    log("")

    train_tf = get_transform(args.img_size, augment=True)
    test_tf  = get_transform(args.img_size, augment=False)
    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.data_dir

    train_ds = HeadPoseDataset(args.data_dir, transform=train_tf, split="train")
    test_ds  = HeadPoseDataset(test_dir,      transform=test_tf,  split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    log(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    log(f"Batches/epoch: train={len(train_loader)}, test={len(test_loader)}\n")

    # resume 처리
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    results  = []
    done     = set()
    if args.resume and os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path)
        results  = df_prev.to_dict("records")
        done     = set(df_prev["model"].tolist())
        print(f"Resume: {len(done)}개 이미 완료\n")

    for name, timm_name in BACKBONES.items():
        if name in done:
            print(f"Skip (already done): {name}")
            continue

        log(f"\n{'='*55}")
        log(f"  Training: {name}")
        log(f"{'='*55}")
        log(f"[Model] Loading {timm_name} (ImageNet pretrained)...")

        model     = HeadPoseModel(timm_name).to(device)
        params    = count_params(model)
        log(f"[Model] Parameters: {params}M")
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_mae  = float("inf")
        ckpt_path = os.path.join(args.output_dir, "checkpoints", f"{name}_best.pth")

        for epoch in range(1, args.epochs + 1):
            t0      = time.time()
            loss    = train_epoch(model, train_loader, optimizer, device, epoch=epoch)
            metrics = evaluate(model, test_loader, device, desc=f"Eval e{epoch}")
            scheduler.step()
            elapsed = time.time() - t0

            if metrics["mean_mae"] < best_mae:
                best_mae = metrics["mean_mae"]
                torch.save(model.state_dict(), ckpt_path)

            log(f"  Epoch {epoch:3d}/{args.epochs} | loss={loss:.4f} | "
                f"Yaw={metrics['yaw_mae']:.2f} "
                f"Pitch={metrics['pitch_mae']:.2f} "
                f"Roll={metrics['roll_mae']:.2f} "
                f"Mean={metrics['mean_mae']:.2f}° | "
                f"best={best_mae:.2f}° | {elapsed:.1f}s")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final   = evaluate(model, test_loader, device)
        fps_gpu = measure_fps(model, device, args.img_size)
        fps_cpu = measure_fps_cpu(model, args.img_size)

        results.append({
            "model":     name,
            "params_M":  params,
            "yaw_mae":   round(final["yaw_mae"],   4),
            "pitch_mae": round(final["pitch_mae"],  4),
            "roll_mae":  round(final["roll_mae"],   4),
            "mean_mae":  round(final["mean_mae"],   4),
            "fps_gpu":   fps_gpu,
            "fps_cpu":   fps_cpu,
        })
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n  ✅ {name} | MAE={final['mean_mae']:.2f}° | "
              f"Params={params}M | CPU={fps_cpu} FPS")

    df = pd.DataFrame(results).sort_values("mean_mae")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*55)
    print("  BASELINE COMPARISON RESULTS")
    print("="*55)
    print(df.to_string(index=False))

    # 시각화
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].bar(df["model"], df["mean_mae"], color="steelblue")
        axes[0].axhline(y=5.0, color="red", linestyle="--", label="Target (5°)")
        axes[0].set_ylabel("Mean MAE (°)")
        axes[0].set_title("Mean MAE Comparison")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].legend()

        axes[1].scatter(df["params_M"], df["mean_mae"], s=120, color="steelblue")
        for _, row in df.iterrows():
            axes[1].annotate(row["model"], (row["params_M"], row["mean_mae"]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes[1].axhline(y=5.0, color="red", linestyle="--", label="Target MAE")
        axes[1].set_xlabel("Parameters (M)")
        axes[1].set_ylabel("Mean MAE (°)")
        axes[1].set_title("Params vs MAE")
        axes[1].legend()

        axes[2].scatter(df["fps_cpu"], df["mean_mae"], s=120, color="steelblue")
        for _, row in df.iterrows():
            axes[2].annotate(row["model"], (row["fps_cpu"], row["mean_mae"]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes[2].axhline(y=5.0,  color="red",  linestyle="--", label="Target MAE")
        axes[2].axvline(x=20.0, color="blue", linestyle="--", label="Target FPS")
        axes[2].set_xlabel("FPS (CPU)")
        axes[2].set_ylabel("Mean MAE (°)")
        axes[2].set_title("FPS vs MAE")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_plot.png"), dpi=150)
        print(f"\n  Plot saved.")
    except Exception as e:
        print(f"  Plot 생성 실패: {e}")

    print(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
