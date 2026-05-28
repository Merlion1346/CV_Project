"""
Experiment 4: Data Augmentation Study
- None  : 증강 없음
- Basic : flip only
- Medium: flip + rotation
- Heavy : flip + rotation + color jitter + blur
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


AUG_CONFIGS = [
    {
        "name":     "None",
        "flip":     False,
        "rotation": False,
        "color":    False,
        "blur":     False,
        "desc":     "No augmentation",
    },
    {
        "name":     "Basic",
        "flip":     True,
        "rotation": False,
        "color":    False,
        "blur":     False,
        "desc":     "Flip only",
    },
    {
        "name":     "Medium",
        "flip":     True,
        "rotation": True,
        "color":    True,
        "blur":     False,
        "desc":     "Flip + Rotation + Color",
    },
    {
        "name":     "Heavy",
        "flip":     True,
        "rotation": True,
        "color":    True,
        "blur":     True,
        "desc":     "Flip + Rotation + Color + Blur",
    },
]


class HeadPoseDataset(Dataset):
    def __init__(self, data_dir, img_size=224, aug_cfg=None, split="train"):
        self.img_size = img_size
        self.aug      = aug_cfg or {}
        self.samples  = []

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

        # 기본 transform
        tf = [transforms.Resize((img_size, img_size))]
        if aug_cfg and aug_cfg.get("color"):
            tf.append(transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                             saturation=0.3, hue=0.1))
        if aug_cfg and aug_cfg.get("blur"):
            tf.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        tf += [transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, yaw, pitch, roll = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.aug.get("flip") and np.random.rand() < 0.5:
            img  = TF.hflip(img)
            yaw  = -yaw
            roll = -roll

        if self.aug.get("rotation"):
            angle = np.random.uniform(-30, 30)
            img   = TF.rotate(img, angle)
            roll  = max(-90, min(90, roll + angle))

        img   = self.transform(img)
        label = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return img, label


class HeadPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/300W_LP")
    parser.add_argument("--test_dir",    default="data/AFLW2000")
    parser.add_argument("--output_dir",  default="results/experiment_4_augmentation")
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

    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.data_dir
    results  = []

    for aug_cfg in AUG_CONFIGS:
        print(f"\n{'='*55}")
        print(f"  Aug: {aug_cfg['name']} — {aug_cfg['desc']}")
        print(f"{'='*55}")

        train_ds = HeadPoseDataset(args.data_dir, args.img_size,
                                   aug_cfg=aug_cfg, split="train")
        test_ds  = HeadPoseDataset(test_dir, args.img_size,
                                   aug_cfg=None, split="test")   # test는 항상 aug 없음

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model     = HeadPoseModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_mae  = float("inf")
        ckpt_path = os.path.join(args.output_dir, f"{aug_cfg['name']}_best.pth")

        for epoch in range(1, args.epochs + 1):
            train_loss    = train_epoch(model, train_loader, optimizer, device)
            train_metrics = evaluate(model, train_loader, device)
            test_metrics  = evaluate(model, test_loader,  device)
            scheduler.step()

            if test_metrics["mean_mae"] < best_mae:
                best_mae = test_metrics["mean_mae"]
                torch.save(model.state_dict(), ckpt_path)

            if epoch % 10 == 0 or epoch == args.epochs:
                gap = test_metrics["mean_mae"] - train_metrics["mean_mae"]
                print(f"  Epoch {epoch:3d} | Train={train_metrics['mean_mae']:.2f}° "
                      f"Test={test_metrics['mean_mae']:.2f}° Gap={gap:+.2f}°")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        train_final = evaluate(model, train_loader, device)
        test_final  = evaluate(model, test_loader,  device)

        results.append({
            "aug":           aug_cfg["name"],
            "desc":          aug_cfg["desc"],
            "train_mae":     round(train_final["mean_mae"], 4),
            "test_yaw_mae":  round(test_final["yaw_mae"],   4),
            "test_pitch_mae":round(test_final["pitch_mae"],  4),
            "test_roll_mae": round(test_final["roll_mae"],   4),
            "test_mean_mae": round(test_final["mean_mae"],   4),
            "generalization_gap": round(test_final["mean_mae"] - train_final["mean_mae"], 4),
        })
        print(f"\n  ✅ {aug_cfg['name']} | Test Mean MAE={test_final['mean_mae']:.2f}°")

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "augmentation_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "="*55)
    print("  AUGMENTATION STUDY RESULTS")
    print("="*55)
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")


if __name__ == "__main__":
    main()
