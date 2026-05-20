"""
Export EfficientNetHeadPose to ONNX for Qualcomm AI Hub deployment.

Usage:
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx --verify
"""

import argparse
import numpy as np
import torch
from model import EfficientNetHeadPose

ANGLE_MAX = 99.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output",     type=str, default="model.onnx")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--variant",    type=str, default=None,
                   help="Override EfficientNet variant (auto-detected from checkpoint)")
    p.add_argument("--verify",     action="store_true",
                   help="Run onnxruntime and compare outputs against PyTorch")
    return p.parse_args()


def load_model(checkpoint_path: str, variant_override: str | None, device: torch.device):
    ckpt    = torch.load(checkpoint_path, map_location=device)
    variant = variant_override or ckpt.get("variant", "b0")
    print(f"[Export] Variant: {variant}  |  Checkpoint: {checkpoint_path}")

    model = EfficientNetHeadPose(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


def export(model, img_size: int, output_path: str):
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        dynamo=False,          # legacy exporter — supports AdaptiveMaxPool2d
        opset_version=17,
        input_names=["image"],
        output_names=["angles_normalized"],
        dynamic_axes={
            "image":             {0: "batch"},
            "angles_normalized": {0: "batch"},
        },
    )
    print(f"[Export] Saved ONNX → {output_path}")
    print(f"         Input : image            (B, 3, {img_size}, {img_size})  — ImageNet-normalized")
    print(f"         Output: angles_normalized (B, 3)  — [yaw, pitch, roll] / {ANGLE_MAX}")
    print(f"         Post-process: multiply output × {ANGLE_MAX} to get degrees")


def verify(model, onnx_path: str, img_size: int):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnxruntime not installed — skipping. pip install onnxruntime")
        return

    dummy  = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess   = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"[Verify] Max abs diff (PyTorch vs ONNX): {max_diff:.6f}  "
          f"({'OK' if max_diff < 1e-4 else 'WARNING — large diff'})")


def main():
    args   = parse_args()
    device = torch.device("cpu")   # export on CPU for portability

    model = load_model(args.checkpoint, args.variant, device)
    export(model, args.img_size, args.output)

    if args.verify:
        verify(model, args.output, args.img_size)


if __name__ == "__main__":
    main()
