"""
Head Pose Estimation — Raspberry Pi Inference (ONNX Runtime)

Requires no PyTorch or torchvision. Runs entirely on CPU with ONNX Runtime.

Setup on Raspberry Pi:
    pip install onnxruntime opencv-python-headless numpy

Usage:
    python inference_rpi.py --model model_int8.onnx
    python inference_rpi.py --model model.onnx --camera 0 --width 640 --height 480
"""

import argparse
import time

import cv2
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────
# Preprocessing (no torchvision needed)
# ─────────────────────────────────────────────
def preprocess(face_bgr: np.ndarray, img_size: int = 224) -> np.ndarray:
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rs   = cv2.resize(face_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    face_norm = (face_rs - IMAGENET_MEAN) / IMAGENET_STD
    return face_norm.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


# ─────────────────────────────────────────────
# Direction label
# ─────────────────────────────────────────────
def angle_to_direction(yaw: float, pitch: float):
    if abs(pitch) >= 15:
        return ("Up",    (220, 220,  0)) if pitch > 0 else ("Down",  (  0, 200, 200))
    if abs(yaw) <= 15:
        return ("Front", (  0, 220,   0))
    return ("Left", (255, 80, 0)) if yaw < 0 else ("Right", (0, 80, 255))


# ─────────────────────────────────────────────
# Axis overlay
# ─────────────────────────────────────────────
def draw_axes(img, yaw, pitch, roll, cx, cy, size=80):
    yr, pr, rr = np.radians(yaw), np.radians(pitch), np.radians(roll)
    Rz = np.array([[np.cos(rr), -np.sin(rr), 0],
                   [np.sin(rr),  np.cos(rr), 0],
                   [0,           0,          1]])
    Ry = np.array([[ np.cos(yr), 0, np.sin(yr)],
                   [0,           1, 0          ],
                   [-np.sin(yr), 0, np.cos(yr)]])
    Rx = np.array([[1, 0,            0         ],
                   [0, np.cos(pr), -np.sin(pr) ],
                   [0, np.sin(pr),  np.cos(pr) ]])
    proj = ((Rz @ Ry @ Rx) @ np.eye(3) * size).T
    orig = np.array([cx, cy])
    for axis, color in zip(range(3), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        end = (orig + proj[axis, :2] * [1, -1]).astype(int)
        cv2.arrowedLine(img, tuple(orig.astype(int)), tuple(end), color, 2, tipLength=0.3)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def run(args):
    import onnxruntime as ort

    sess = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"],
    )
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"[RPi] Model loaded: {args.model}")
    print(f"[RPi] Input: {input_name}  Output: {output_name}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[RPi] Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("[RPi] Press 'q' to quit.")

    ema_angles = None
    t_prev     = time.perf_counter()

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
            x1  = max(0, x - pad);              y1 = max(0, y - pad)
            x2  = min(frame.shape[1], x+w+pad); y2 = min(frame.shape[0], y+h+pad)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            inp     = preprocess(face, args.img_size)
            out_deg = sess.run([output_name], {input_name: inp})[0][0]  # (3,) degrees

            # EMA smoothing
            if ema_angles is None:
                ema_angles = out_deg.copy()
            else:
                ema_angles = args.ema_alpha * out_deg + (1 - args.ema_alpha) * ema_angles
            yaw, pitch, roll = ema_angles

            direction, color = angle_to_direction(yaw, pitch)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, direction,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)
            cv2.putText(frame,
                        f"Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  Roll:{roll:+.1f}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            draw_axes(frame, yaw, pitch, roll,
                      x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2,
                      size=min(w, h) // 3)

        # FPS
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        cv2.imshow("Head Pose — RPi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Head Pose — Raspberry Pi Inference (ONNX)")
    p.add_argument("--model",     type=str,   required=True,
                   help="Path to model.onnx or model_int8.onnx")
    p.add_argument("--img_size",  type=int,   default=224)
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--width",     type=int,   default=640)
    p.add_argument("--height",    type=int,   default=480)
    p.add_argument("--ema_alpha", type=float, default=0.2,
                   help="EMA smoothing factor (0.1–0.4 recommended)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
