"""
EfficientNet Head Pose — Real-time Webcam Inference (Regression Only)
Usage:
    python inference.py --checkpoint ./checkpoints/best.pth
"""

import argparse
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import EfficientNetHeadPose


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_ANGLE_MAX    = np.array([90.0, 90.0, 90.0])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(angles_np: np.ndarray) -> np.ndarray:
    """Convert model output [-1, 1] → degrees."""
    return angles_np * _ANGLE_MAX


def angle_to_direction(yaw: float, pitch: float):
    """
    yaw/pitch 값으로 시선 방향 문자열과 BGR 색상을 반환합니다.
    분류 헤드 없이 회귀값으로 방향을 결정합니다.
    """
    if abs(pitch) >= 15:
        return ("Up",    (220, 220,  0)) if pitch > 0 else ("Down",  ( 0, 200, 200))
    if abs(yaw) <= 15:
        return ("Front", ( 0, 220,   0))
    return ("Left",  (255,  80,  0)) if yaw < 0 else ("Right", ( 0,  80, 255))


def draw_axes(img, yaw, pitch, roll, cx, cy, size=80):
    """Overlay 3D orientation axes centred at (cx, cy)."""
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
    for axis, color in zip(range(3), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        end = (orig + proj[axis, :2] * [1, -1]).astype(int)
        cv2.arrowedLine(img, tuple(orig.astype(int)), tuple(end),
                        color, 2, tipLength=0.3)


# ─────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────
@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    # ── Load checkpoint ───────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    var  = ckpt.get("variant", args.variant)
    print(f"[Inference] Variant: {var}")

    # ── Load model ────────────────────────────
    model = EfficientNetHeadPose(variant=var, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform    = build_transform(args.img_size)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Inference] Resolution: {actual_w}x{actual_h}")
    print("[Inference] Press 'q' to quit.")

    # ── EMA smoothing 상태 ────────────────────
    ema_angles = None   # 첫 프레임에서 초기화

    fps_tick = cv2.getTickCount()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            pad = int(0.15 * min(w, h))
            x1  = max(0, x - pad);             y1 = max(0, y - pad)
            x2  = min(frame.shape[1], x+w+pad); y2 = min(frame.shape[0], y+h+pad)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ── Inference ─────────────────────
            tensor = transform(
                Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(device)

            pred = model(tensor)   # (1, 3)  — normalized [yaw, pitch, roll]

            # ── EMA angle smoothing ────────────
            raw_angles = denormalize(pred[0].cpu().numpy())
            if ema_angles is None:
                ema_angles = raw_angles.copy()
            else:
                ema_angles = (args.ema_alpha * raw_angles
                              + (1 - args.ema_alpha) * ema_angles)
            yaw, pitch, roll = ema_angles

            # ── 방향 결정 (회귀값 기반) ────────
            direction, color = angle_to_direction(yaw, pitch)

            # ── Draw ──────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, direction,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)
            cv2.putText(frame,
                        f"Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  Roll:{roll:+.1f}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            draw_axes(frame, yaw, pitch, roll,
                      x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2,
                      size=min(w, h) // 3)

        # ── FPS overlay ───────────────────────
        fps      = cv2.getTickFrequency() / (cv2.getTickCount() - fps_tick)
        fps_tick = cv2.getTickCount()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        cv2.imshow("Head Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Head Pose — Real-time Inference (Regression Only)")
    p.add_argument("--checkpoint", type=str,   required=True)
    p.add_argument("--variant",    type=str,   default="b0",
                   choices=["b0","b1","b2","b3","b4","b5","b6","b7"])
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--width",      type=int,   default=1280)
    p.add_argument("--height",     type=int,   default=720)
    p.add_argument("--ema_alpha",  type=float, default=0.2,
                   help="EMA 스무딩 계수 (낮을수록 부드러움, 권장 0.1~0.4)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
