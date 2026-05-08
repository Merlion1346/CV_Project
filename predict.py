"""
Head Pose Prediction Visualizer (Regression Only)
Runs model on validation/test set and saves annotated images.

Usage:
    python predict.py --checkpoint ./checkpoints/best.pth --data_dir kface_data
    python predict.py --checkpoint ./checkpoints/best.pth --data_dir kface_data --num_samples 100 --split test
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

from dataset import build_dataframe, get_transforms
from model   import EfficientNetHeadPose


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ANGLE_MAX = np.array([90.0, 90.0, 90.0])


def denormalize_angles(angles_norm: torch.Tensor) -> np.ndarray:
    return (angles_norm.cpu().numpy() * ANGLE_MAX)


def angle_to_direction(yaw: float, pitch: float) -> str:
    if abs(pitch) >= 15:
        return "up" if pitch > 0 else "down"
    if abs(yaw) <= 15:
        return "front"
    return "left" if yaw < 0 else "right"


DIRECTION_COLOR = {
    "front": ( 0, 200,   0),
    "left":  (255,  80,   0),
    "right": (  0,  80, 255),
    "up":    (220, 220,   0),
    "down":  (  0, 200, 200),
}


# ─────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────
def draw_axes(img, yaw, pitch, roll, cx, cy, size=60):
    yr, pr, rr = np.radians(yaw), np.radians(pitch), np.radians(roll)
    Rz = np.array([[np.cos(rr), -np.sin(rr), 0],
                   [np.sin(rr),  np.cos(rr), 0],
                   [0,           0,          1]])
    Ry = np.array([[ np.cos(yr), 0, np.sin(yr)],
                   [0,           1, 0          ],
                   [-np.sin(yr), 0, np.cos(yr)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(pr), -np.sin(pr) ],
                   [0, np.sin(pr),  np.cos(pr) ]])
    proj = ((Rz @ Ry @ Rx) @ np.eye(3) * size).T
    orig = np.array([cx, cy])
    for axis, color in zip(range(3), [(0,0,255),(0,255,0),(255,0,0)]):
        end = (orig + proj[axis, :2] * np.array([1, -1])).astype(int)
        cv2.arrowedLine(img, tuple(orig.astype(int)), tuple(end),
                        color, 2, tipLength=0.3)


def annotate_image(img_bgr, pred_angles, true_angles, pred_dir, true_dir, correct):
    """
    회귀 결과를 이미지에 시각화합니다.
    초록 테두리 = 방향 일치, 빨간 테두리 = 방향 불일치.
    """
    h, w = img_bgr.shape[:2]
    out  = img_bgr.copy()

    border_color = (0, 200, 0) if correct else (0, 0, 220)
    cv2.rectangle(out, (0, 0), (w-1, h-1), border_color, 4)

    pred_color = DIRECTION_COLOR.get(pred_dir, (255, 255, 255))
    true_color = (200, 200, 200)

    py, pp, pr = pred_angles
    ty, tp, tr = true_angles

    # 예측 방향 + 각도
    cv2.putText(out, f"Pred: {pred_dir}",
                (6, 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, pred_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"GT:   {true_dir}",
                (6, 42), cv2.FONT_HERSHEY_DUPLEX, 0.55, true_color, 1, cv2.LINE_AA)

    # 각도 수치
    cv2.putText(out, f"P Yaw:{py:+.1f} Pit:{pp:+.1f}",
                (6, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"T Yaw:{ty:+.1f} Pit:{tp:+.1f}",
                (6, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, true_color, 1, cv2.LINE_AA)

    # MAE 표시
    mae = np.abs(pred_angles - true_angles).mean()
    cv2.putText(out, f"MAE:{mae:.1f}deg",
                (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    draw_axes(out, py, pp, pr, w//2, h//2, size=min(w, h)//5)
    return out


# ─────────────────────────────────────────────
# Predict & Visualize
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_and_save(model, df, device, use_amp,
                     img_size, num_samples, output_dir, grid_cols):

    transform = get_transforms("val", img_size)

    if num_samples and num_samples < len(df):
        sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        sample_df = df.reset_index(drop=True)

    # 방향별 서브 디렉토리
    for direction in DIRECTION_COLOR:
        os.makedirs(os.path.join(output_dir, "correct",   direction), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "incorrect", direction), exist_ok=True)

    results   = []
    cell_size = img_size

    iterable = (tqdm(sample_df.iterrows(), total=len(sample_df), desc="Predicting")
                if USE_TQDM else sample_df.iterrows())

    for idx, row in iterable:
        img_pil = Image.open(row["path"]).convert("RGB")
        tensor  = transform(img_pil).unsqueeze(0).to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(tensor)   # (1, 3)

        pred_angles = denormalize_angles(pred[0])          # [yaw, pitch, roll]
        true_angles = np.array([row["yaw"], row["pitch"], row["roll"]], dtype=float)

        pred_dir = angle_to_direction(pred_angles[0], pred_angles[1])
        true_dir = angle_to_direction(true_angles[0], true_angles[1])
        correct  = (pred_dir == true_dir)

        img_bgr   = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_bgr   = cv2.resize(img_bgr, (cell_size, cell_size))
        annotated = annotate_image(img_bgr, pred_angles, true_angles,
                                   pred_dir, true_dir, correct)

        folder    = "correct" if correct else "incorrect"
        mae_str   = f"{np.abs(pred_angles - true_angles).mean():.1f}"
        filename  = f"{idx:05d}_{true_dir}_pred{pred_dir}_mae{mae_str}.jpg"
        cv2.imwrite(os.path.join(output_dir, folder, true_dir, filename), annotated)

        results.append({
            "img":         annotated,
            "correct":     correct,
            "pred_dir":    pred_dir,
            "true_dir":    true_dir,
            "pred_angles": pred_angles,
            "true_angles": true_angles,
        })

    # ── Summary grid ──────────────────────────
    print("\n[Predict] Building summary grid...")
    cols   = grid_cols
    rows   = (len(results) + cols - 1) // cols
    grid   = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for i, r in enumerate(results):
        ry, cx = divmod(i, cols)
        grid[ry*cell_size:(ry+1)*cell_size, cx*cell_size:(cx+1)*cell_size] = r["img"]

    grid_path = os.path.join(output_dir, "summary_grid.jpg")
    cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── Stats ─────────────────────────────────
    total     = len(results)
    correct_n = sum(r["correct"] for r in results)
    all_mae   = np.stack([np.abs(r["pred_angles"] - r["true_angles"]) for r in results])

    print(f"\n{'='*52}")
    print(f"  Total   : {total}")
    print(f"  Correct : {correct_n} ({correct_n/total*100:.1f}%)  "
          f"Incorrect: {total-correct_n} ({(total-correct_n)/total*100:.1f}%)")
    print(f"  MAE     : Yaw={all_mae[:,0].mean():.2f}°  "
          f"Pitch={all_mae[:,1].mean():.2f}°  "
          f"Roll={all_mae[:,2].mean():.2f}°  "
          f"Mean={all_mae.mean():.2f}°")
    print(f"{'='*52}")

    class_total   = defaultdict(int)
    class_correct = defaultdict(int)
    class_mae     = defaultdict(list)
    for r in results:
        td = r["true_dir"]
        class_total[td]   += 1
        class_mae[td].append(np.abs(r["pred_angles"] - r["true_angles"]).mean())
        if r["correct"]:
            class_correct[td] += 1

    print(f"\n  {'Direction':<10} {'Correct':>8} {'Total':>7} {'Acc':>7} {'MAE':>8}")
    print(f"  {'-'*46}")
    for cls in sorted(class_total):
        t = class_total[cls]
        c = class_correct[cls]
        m = np.mean(class_mae[cls])
        print(f"  {cls:<10} {c:>8} {t:>7} {c/t*100:>6.1f}% {m:>7.2f}°")

    print(f"\n[Predict] Images  → {output_dir}/correct|incorrect/")
    print(f"[Predict] Grid    → {grid_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[Predict] Device: {device}")

    ckpt    = torch.load(args.checkpoint, map_location=device)
    variant = ckpt.get("variant", args.variant)
    print(f"[Predict] Variant: {variant} | Checkpoint: {args.checkpoint}")

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
    print(f"[Predict] Split: {args.split} | Samples: {len(target_df)}")

    predict_and_save(
        model       = model,
        df          = target_df,
        device      = device,
        use_amp     = use_amp,
        img_size    = args.img_size,
        num_samples = args.num_samples,
        output_dir  = args.output_dir,
        grid_cols   = args.grid_cols,
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Predict & Visualize — Regression Only")
    p.add_argument("--checkpoint",  type=str,   required=True)
    p.add_argument("--data_dir",    type=str,   required=True)
    p.add_argument("--output_dir",  type=str,   default="./predictions")
    p.add_argument("--split",       type=str,   default="val", choices=["val", "test"])
    p.add_argument("--variant",     type=str,   default="b0",
                   choices=["b0","b1","b2","b3","b4","b5","b6","b7"])
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--num_samples", type=int,   default=100,
                   help="시각화할 샘플 수 (0=전체)")
    p.add_argument("--grid_cols",   type=int,   default=10)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
