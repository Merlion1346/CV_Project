"""
Export EfficientNetHeadPose to ONNX (with baked-in soft-argmax).

The exported model accepts an ImageNet-normalized image and directly outputs
angles in degrees — no post-processing required at inference time.

Usage:
    # FP32 export
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx

    # FP32 export + verify
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx --verify

    # INT8 static quantization (QDQ format, required for Raspberry Pi / aarch64)
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx \\
        --quantize --calib_dir data/AFLW2000 --n_calib 200
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from model import EfficientNetHeadPose, HeadPoseLoss, N_BINS


# ─────────────────────────────────────────────
# Wrapper: bake soft-argmax into the graph
# ─────────────────────────────────────────────
class HeadPoseExportable(nn.Module):
    """Wraps model + HeadPoseLoss.predict so ONNX output is degrees directly."""

    def __init__(self, backbone: EfficientNetHeadPose, loss: HeadPoseLoss):
        super().__init__()
        self.backbone = backbone
        self.loss     = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)          # (B, 3, N_BINS)
        return self.loss.predict(logits)   # (B, 3)  — [yaw, pitch, roll] in degrees


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str, variant_override, device):
    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    variant = variant_override or ckpt.get("variant", "b0")
    print(f"[Export] Variant: {variant}  |  Checkpoint: {checkpoint_path}")

    backbone = EfficientNetHeadPose(variant=variant, pretrained=False)
    backbone.load_state_dict(ckpt["model"])
    backbone.eval()

    loss = HeadPoseLoss(n_bins=N_BINS)
    return HeadPoseExportable(backbone, loss).to(device)


def export(model, img_size: int, output_path: str):
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        dynamo=False,
        opset_version=17,
        input_names=["image"],
        output_names=["angles_deg"],
        dynamic_axes={
            "image":      {0: "batch"},
            "angles_deg": {0: "batch"},
        },
    )
    print(f"[Export] Saved ONNX → {output_path}")
    print(f"         Input : image      (B, 3, {img_size}, {img_size})  — ImageNet-normalized")
    print(f"         Output: angles_deg (B, 3)  — [yaw, pitch, roll] in degrees")


class FaceImageCalibReader:
    """Calibration data reader using real face images.

    Applies the same preprocessing as HeadPosePredictor so that scale/zero-point
    values are calibrated on the actual input distribution.
    """

    def __init__(self, image_dir: str, n: int = 200):
        import cv2
        import os
        self._batches = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        fnames = [f for f in os.listdir(image_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n]
        if not fnames:
            raise ValueError(f"[Quantize] No images found in {image_dir}")

        for fname in fnames:
            img = cv2.imread(os.path.join(image_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)[np.newaxis]   # HWC → 1CHW
            self._batches.append({"image": img})

        print(f"[Quantize] Calibration set: {len(self._batches)} images from {image_dir}")
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._batches):
            return None
        batch = self._batches[self._idx]
        self._idx += 1
        return batch


def quantize(fp32_path: str, int8_path: str, calib_dir: str, n_calib: int = 200):
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantType, QuantFormat, quant_pre_process,
        )
    except ImportError:
        print("[Quantize] onnxruntime not installed — pip install onnxruntime")
        return

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        prep_path = f.name

    try:
        print(f"[Quantize] Pre-processing model for shape inference …")
        quant_pre_process(fp32_path, prep_path)

        reader = FaceImageCalibReader(calib_dir, n=n_calib)

        # QDQ format: inserts QuantizeLinear/DequantizeLinear nodes — fully supported
        # on ONNX Runtime aarch64 CPU EP (unlike ConvInteger/MatMulInteger).
        quantize_static(
            prep_path,
            int8_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,          # per_channel=True is slower on aarch64
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
    finally:
        os.unlink(prep_path)

    print(f"[Quantize] INT8 (QDQ) model saved → {int8_path}")


def verify(model, onnx_path: str, img_size: int):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnxruntime not installed — pip install onnxruntime")
        return

    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"[Verify] Max abs diff PyTorch vs ONNX: {max_diff:.4f}°  "
          f"({'OK' if max_diff < 0.1 else 'WARNING — large diff'})")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,  required=True)
    p.add_argument("--output",     type=str,  default="models/model.onnx")
    p.add_argument("--img_size",   type=int,  default=224)
    p.add_argument("--variant",    type=str,  default=None,
                   help="Override EfficientNet variant (auto-detected from checkpoint)")
    p.add_argument("--verify",     action="store_true",
                   help="Compare ONNX output against PyTorch")
    p.add_argument("--quantize",   action="store_true",
                   help="Export INT8 QDQ-quantized model (required for Raspberry Pi / aarch64)")
    p.add_argument("--calib_dir",  type=str,  default=None,
                   help="Directory of face images for static quantization calibration")
    p.add_argument("--n_calib",    type=int,  default=200,
                   help="Number of calibration images (default: 200)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu")

    model = load_model(args.checkpoint, args.variant, device)
    export(model, args.img_size, args.output)

    if args.verify:
        verify(model, args.output, args.img_size)

    if args.quantize:
        if not args.calib_dir:
            raise SystemExit("[Quantize] --calib_dir is required for static quantization")
        int8_path = args.output.replace(".onnx", "_int8.onnx")
        quantize(args.output, int8_path, args.calib_dir, args.n_calib)
        if args.verify:
            verify(model, int8_path, args.img_size)


if __name__ == "__main__":
    main()
