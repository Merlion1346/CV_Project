"""
Export EfficientNetHeadPose to ONNX (with baked-in soft-argmax).

The exported model accepts an ImageNet-normalized image and directly outputs
angles in degrees — no post-processing required at inference time.

Usage:
    # FP32 export
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx

    # FP32 export + verify
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx --verify

    # INT8 dynamic quantization (recommended for Raspberry Pi)
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx --quantize
"""

import argparse
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


def quantize(fp32_path: str, int8_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[Quantize] onnxruntime not installed — pip install onnxruntime")
        return

    # Conv → ConvInteger is not implemented in ONNX Runtime CPU EP on ARM.
    # Quantize only MatMul/Gemm (attention/FC layers) and skip Conv entirely.
    quantize_dynamic(
        fp32_path, int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"[Quantize] INT8 model saved → {int8_path}")


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
    p.add_argument("--output",     type=str,  default="model.onnx")
    p.add_argument("--img_size",   type=int,  default=224)
    p.add_argument("--variant",    type=str,  default=None,
                   help="Override EfficientNet variant (auto-detected from checkpoint)")
    p.add_argument("--verify",     action="store_true",
                   help="Compare ONNX output against PyTorch")
    p.add_argument("--quantize",   action="store_true",
                   help="Export INT8 dynamic-quantized model (recommended for Raspberry Pi)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cpu")

    model = load_model(args.checkpoint, args.variant, device)
    export(model, args.img_size, args.output)

    if args.verify:
        verify(model, args.output, args.img_size)

    if args.quantize:
        int8_path = args.output.replace(".onnx", "_int8.onnx")
        quantize(args.output, int8_path)
        if args.verify:
            verify(model, int8_path, args.img_size)


if __name__ == "__main__":
    main()
