"""
Stable Diffusion Inpainting — Hat Augmentation for 300W-LP

이마/머리 위를 마스킹한 뒤 SD Inpainting으로 모자를 합성합니다.
생성된 이미지는 원본 .mat(pose label)을 그대로 복사해 학습에 바로 사용 가능합니다.

Usage:
    python scripts/augment_hats_sd.py \\
        --data_dir  /path/to/300W_LP \\
        --output_dir ./data/hat_augmented \\
        --num_per_image 1 \\
        --steps 30
"""

import os
import argparse
import random
import shutil
from glob import glob

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline

# ─────────────────────────────────────────────
# Prompts / Negative
# ─────────────────────────────────────────────
HAT_PROMPTS = [
    "person wearing a baseball cap, photorealistic, natural lighting",
    "person wearing a beanie hat, photorealistic, natural lighting",
    "person wearing a knit winter hat, photorealistic, natural lighting",
    "person wearing a snapback cap, photorealistic, natural lighting",
    "person wearing a bucket hat, photorealistic, natural lighting",
]

NEGATIVE_PROMPT = (
    "blurry, deformed face, bad anatomy, extra limbs, mutation, "
    "lowres, watermark, text, signature, cropped, worst quality"
)


# ─────────────────────────────────────────────
# Landmark / Mask
# ─────────────────────────────────────────────
def load_landmarks(mat_path: str):
    """300W-LP .mat → (68, 2) 2D landmarks (x, y). 없으면 None."""
    try:
        mat = sio.loadmat(mat_path)
        if "pt2d" in mat:
            return mat["pt2d"].T  # (2, 68) → (68, 2)
    except Exception:
        pass
    return None


def create_hat_mask(inpaint_size: int, orig_w: int, orig_h: int,
                    landmarks=None) -> Image.Image:
    """
    이마 위~머리 꼭대기 영역을 흰색으로 마스킹.
    landmarks가 있으면 눈썹 y좌표 기준으로 계산, 없으면 상단 40%.
    """
    mask = Image.new("L", (inpaint_size, inpaint_size), 0)
    draw = ImageDraw.Draw(mask)

    if landmarks is not None:
        # 원본 이미지 스케일 → inpaint_size 스케일 변환
        scale_y = inpaint_size / orig_h
        scale_x = inpaint_size / orig_w

        lm = landmarks.copy().astype(float)
        lm[:, 0] *= scale_x
        lm[:, 1] *= scale_y

        # 눈썹 랜드마크: 68점 기준 17~26번
        brow_y = lm[17:27, 1].min()
        # 눈썹보다 약간 아래까지 마스크 (모자 챙 포함)
        bottom = int(brow_y + inpaint_size * 0.05)
        bottom = max(bottom, int(inpaint_size * 0.20))   # 최소 20%
        bottom = min(bottom, int(inpaint_size * 0.55))   # 최대 55%
    else:
        bottom = int(inpaint_size * 0.40)

    draw.rectangle([0, 0, inpaint_size, bottom], fill=255)

    # 경계 부드럽게 처리 (자연스러운 합성을 위해)
    return mask.filter(ImageFilter.GaussianBlur(radius=6))


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
def load_pipeline(model_id: str, device: str) -> StableDiffusionInpaintPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[Pipeline] xformers enabled")
    except Exception:
        pass
    return pipe


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def process(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    print(f"[Pipeline] Loading {args.model_id} ...")
    pipe = load_pipeline(args.model_id, device)

    img_paths = sorted(glob(os.path.join(args.data_dir, "**", "*.jpg"), recursive=True))
    print(f"[Data] Found {len(img_paths)} images in {args.data_dir}")

    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]
        print(f"[Data] Limited to {len(img_paths)} images (--max_images)")

    os.makedirs(args.output_dir, exist_ok=True)

    generated, skipped = 0, 0
    for img_path in img_paths:
        mat_path = os.path.splitext(img_path)[0] + ".mat"
        if not os.path.exists(mat_path):
            skipped += 1
            continue

        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        orig_w, orig_h = orig.size
        landmarks = load_landmarks(mat_path)

        # SD Inpainting은 512x512 또는 768x768 권장
        inpaint_size = args.inpaint_size
        img_resized  = orig.resize((inpaint_size, inpaint_size), Image.LANCZOS)
        mask         = create_hat_mask(inpaint_size, orig_w, orig_h, landmarks)

        for i in range(args.num_per_image):
            prompt = random.choice(HAT_PROMPTS)

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=img_resized,
                    mask_image=mask,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=inpaint_size,
                    width=inpaint_size,
                ).images[0]

            # 저장 경로: output_dir 아래에 원본과 동일한 폴더 구조 유지
            rel      = os.path.relpath(img_path, args.data_dir)
            stem     = os.path.splitext(rel)[0]
            out_img  = os.path.join(args.output_dir, f"{stem}_hat{i}.jpg")
            out_mat  = os.path.join(args.output_dir, f"{stem}_hat{i}.mat")

            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            result.save(out_img, quality=95)
            shutil.copy(mat_path, out_mat)

            generated += 1

        if generated % 50 == 0 and generated > 0:
            print(f"  [Progress] {generated} generated | {skipped} skipped")

    print(f"\n[Done] Generated: {generated} | Skipped: {skipped}")
    print(f"[Done] Output: {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="SD Inpainting hat augmentation for 300W-LP"
    )
    p.add_argument("--data_dir",       type=str, required=True,
                   help="300W_LP 루트 디렉토리")
    p.add_argument("--output_dir",     type=str, default="./data/hat_augmented",
                   help="생성 이미지 저장 경로 (train.py의 --data_dir에 추가 가능)")
    p.add_argument("--model_id",       type=str,
                   default="stabilityai/stable-diffusion-2-inpainting",
                   help="HuggingFace 모델 ID")
    p.add_argument("--num_per_image",  type=int, default=1,
                   help="이미지 1장당 생성할 augmented 이미지 수")
    p.add_argument("--max_images",     type=int, default=0,
                   help="처리할 최대 이미지 수 (0 = 전체)")
    p.add_argument("--steps",          type=int, default=30,
                   help="Denoising step 수 (많을수록 품질↑, 속도↓)")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Classifier-free guidance scale")
    p.add_argument("--inpaint_size",   type=int, default=512,
                   help="SD 입력 해상도 (512 권장)")
    return p.parse_args()


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    process(args)
