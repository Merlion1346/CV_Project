"""
Experiment 5: Input Resolution Study
- 128×128 / 160×160 / 192×192 / 224×224 / 256×256
- 정확도 vs 추론 속도 trade-off 분석
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


RESOLUTIONS = [128, 160, 192, 224, 256]


class HeadPoseDataset(Dataset):
    def __init__(self, data_dir, img_size=224, augment=True, split="train"):
        self.samples = []

        for mat_path in glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True):
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

        tf = [transforms.Resize((img_size, img_size))]
        if augment:
            tf += [transforms.RandomHorizontalFlip(0.5),
                   transforms.ColorJitter(0.2, 0.2, 0.2)]
        tf += [transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, yaw, pitch, roll = self.samples[idx]
        img   = self.transform(Image.open(img_path).convert("RGB"))
        label = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return img, label


class HeadPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    for imgs, labels in loader:
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


def measure_fps(model, device, img_size, n_iter=200):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    for _ in range(10):
        model(dummy)
    t = time.time()
    for _ in range(n_iter):
        model(dummy)
    return round(n_iter / (time.time() - t), 1)


def measure_fps_cpu(model, img_size, n_iter=100):
    model_cpu = model.cpu().eval()
    dummy     = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        model_cpu(dummy)
    t = time.time()
    for _ in range(n_iter):
        model_cpu(dummy)
    return round(n_iter / (time.time() - t), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/300W_LP")
    parser.add_argument("--test_dir",    default="data/AFLW2000")
    parser.add_argument("--output_dir",  default="results/experiment_5_resolution")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--quick_test",  action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 3

    os.makedirs(args.output_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.data_dir
    results  = []

    for img_size in RESOLUTIONS:
        print(f"\n{'='*55}")
        print(f"  Resolution: {img_size}×{img_size}")
        print(f"{'='*55}")

        train_ds = HeadPoseDataset(args.data_dir, img_size=img_size,
                                   augment=True,  split="train")
        test_ds  = HeadPoseDataset(test_dir,      img_size=img_size,
                                   augment=False, split="test")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model     = HeadPoseModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_mae  = float("inf")
        ckpt_path = os.path.join(args.output_dir, f"res{img_size}_best.pth")

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            metrics = evaluate(model, test_loader, device)
            scheduler.step()

            if metrics["mean_mae"] < best_mae:
                best_mae = metrics["mean_mae"]
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d} | Mean MAE={metrics['mean_mae']:.2f}°")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final   = evaluate(model, test_loader, device)
        fps_gpu = measure_fps(model, device, img_size)
        fps_cpu = measure_fps_cpu(model, img_size)

        results.append({
            "resolution":  f"{img_size}×{img_size}",
            "img_size":    img_size,
            "yaw_mae":     round(final["yaw_mae"],   4),
            "pitch_mae":   round(final["pitch_mae"],  4),
            "roll_mae":    round(final["roll_mae"],   4),
            "mean_mae":    round(final["mean_mae"],   4),
            "fps_gpu":     fps_gpu,
            "fps_cpu":     fps_cpu,
        })
        print(f"\n  ✅ {img_size}×{img_size} | MAE={final['mean_mae']:.2f}° "
              f"| GPU={fps_gpu} FPS | CPU={fps_cpu} FPS")

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "resolution_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*55)
    print("  RESOLUTION STUDY RESULTS")
    print("="*55)
    print(df.to_string(index=False))

    # 시각화
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(df["img_size"], df["mean_mae"], "bo-", linewidth=2)
        ax1.axhline(y=5.0, color="red", linestyle="--", label="Target MAE (5°)")
        ax1.set_xlabel("Input Size (px)")
        ax1.set_ylabel("Mean MAE (°)")
        ax1.set_title("Resolution vs Accuracy")
        ax1.legend()

        ax2.plot(df["img_size"], df["fps_cpu"], "go-", linewidth=2, label="CPU FPS")
        ax2.plot(df["img_size"], df["fps_gpu"], "bo-", linewidth=2, label="GPU FPS")
        ax2.axhline(y=20, color="red", linestyle="--", label="Target FPS (20)")
        ax2.set_xlabel("Input Size (px)")
        ax2.set_ylabel("FPS")
        ax2.set_title("Resolution vs Speed")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "resolution_tradeoff.png"), dpi=150)
        print(f"\n  Plot saved.")
    except Exception as e:
        print(f"  Plot 생성 실패: {e}")

    print(f"\n  Saved: {csv_path}")


if __name__ == "__main__":
    main()
