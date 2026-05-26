"""
EfficientNet Head Pose → Core ML (.mlpackage) 변환 스크립트

사전 요구사항:
    pip install coremltools>=7.0

사용법:
    python convert_to_coreml.py --checkpoint checkpoints/260526_b0_soft_argmax_spatial_axis-weighted/best.pth
    python convert_to_coreml.py --checkpoint checkpoints/best.pth --output HeadPose.mlpackage

출력:
    HeadPose.mlpackage — iOS 앱에 추가할 Core ML 모델
    입력:  image (RGB, 224×224, ImageNet 정규화 자동 적용)
    출력:  angles_deg (float32[3]) — [yaw, pitch, roll] 단위: 도(°)
"""

import argparse
import torch
import torch.nn as nn
import coremltools as ct

from model import EfficientNetHeadPose, HeadPoseLoss, N_BINS


# ─────────────────────────────────────────────
# Export wrapper: soft-argmax 내장, batch dim 제거
# ─────────────────────────────────────────────
class HeadPoseForCoreML(nn.Module):
    """
    Core ML / VNCoreMLRequest 호환 래퍼.
    - 입력: (1, 3, 224, 224) — ImageNet 정규화된 float
    - 출력: (3,)            — [yaw, pitch, roll] in degrees (batch dim 제거)
    """
    def __init__(self, backbone: EfficientNetHeadPose, loss: HeadPoseLoss):
        super().__init__()
        self.backbone = backbone
        self.loss     = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)         # (1, 3, N_BINS)
        angles = self.loss.predict(logits) # (1, 3)
        return angles.squeeze(0)           # (3,)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str, variant_override: str | None) -> HeadPoseForCoreML:
    ckpt    = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = variant_override or ckpt.get("variant", "b0")
    print(f"[Convert] Variant: {variant}  |  Checkpoint: {checkpoint_path}")

    backbone = EfficientNetHeadPose(variant=variant, pretrained=False)
    backbone.load_state_dict(ckpt["model"])
    backbone.eval()

    loss = HeadPoseLoss(n_bins=N_BINS)
    return HeadPoseForCoreML(backbone, loss).eval()


def convert(model: HeadPoseForCoreML, img_size: int, output_path: str) -> None:
    dummy  = torch.randn(1, 3, img_size, img_size)
    traced = torch.jit.trace(model, dummy)
    print("[Convert] TorchScript trace 완료")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, img_size, img_size),
            color_layout=ct.colorlayout.RGB,
            mean=[0.485, 0.456, 0.406],   # ImageNet mean (0–1 scale)
            std=[0.229, 0.224, 0.225],    # ImageNet std  (0–1 scale)
        )],
        outputs=[ct.TensorType(name="angles_deg", dtype=float)],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # 메타데이터
    mlmodel.short_description = "EfficientNet Head Pose Estimation"
    mlmodel.input_description["image"]      = "얼굴 이미지 (224×224 RGB)"
    mlmodel.output_description["angles_deg"] = "[yaw, pitch, roll] 단위: 도(°)"

    mlmodel.save(output_path)
    print(f"[Convert] 저장 완료 → {output_path}")
    print(f"          입력: image (RGB 224×224, ImageNet 정규화 자동 적용)")
    print(f"          출력: angles_deg [yaw, pitch, roll] in degrees")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="학습된 체크포인트 경로 (예: checkpoints/best.pth)")
    p.add_argument("--output",     default="HeadPose.mlpackage",
                   help="출력 Core ML 패키지 경로 (기본: HeadPose.mlpackage)")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--variant",    default=None,
                   help="EfficientNet variant 오버라이드 (체크포인트에서 자동 감지)")
    return p.parse_args()


def main():
    args  = parse_args()
    model = load_model(args.checkpoint, args.variant)
    convert(model, args.img_size, args.output)
    print("\n[다음 단계]")
    print("  1. HeadPose.mlpackage 를 Mac으로 복사")
    print("  2. Xcode 프로젝트에 HeadPose.mlpackage 드래그앤드롭")
    print("  3. 앱 빌드 & 실행")


if __name__ == "__main__":
    main()
