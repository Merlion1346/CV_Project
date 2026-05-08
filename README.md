# Head Pose Estimation

EfficientNet-based head pose estimation that predicts **yaw / pitch / roll** from a single face image.  
Pretrained on 300W-LP, fine-tuned on AIHub KFace (Dataset #83).

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 1. Pretrain on 300W-LP
python train_300wlp.py --data_dir /path/to/300W_LP --variant b0 --epochs 50

# 2. Fine-tune on KFace
python train.py --data_dir /path/to/kface_data --variant b0 --epochs 30 \
  --pretrained_ckpt checkpoints/best_300wlp.pth

# 3. Evaluate
python evaluate.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data

# 4. Live webcam demo
python inference.py --checkpoint checkpoints/best.pth
```

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNet B0–B7 (ImageNet pretrained) |
| Attention | Channel Attention (lightweight CBAM) |
| Head | Linear 512 → 128 → 3 · BN · SiLU · Dropout |
| Output | Normalized [−1, 1] → ×90° = degrees |
| Loss | MSE on normalized (yaw, pitch, roll) |

---

## Training

Two-phase schedule applies automatically:

| Phase | Condition | What trains |
|---|---|---|
| 1 — Warm-up | epochs 1 → `warmup_epochs` | Head + attention only |
| 2 — Fine-tune | `warmup_epochs + 1` → end | + top 3 backbone blocks (LR ×0.05) |

### Hyperparameters

| Argument | Default | Notes |
|---|---|---|
| `--variant` | `b0` | b0–b7; larger = better accuracy, more VRAM |
| `--img_size` | `224` | Recommended: 380 for b4+, 456 for b5+ |
| `--batch_size` | `64` | Reduce to 32 for b5+ on 8 GB VRAM |
| `--lr` | `3e-4` | AdamW base LR |
| `--warmup_epochs` | `5` | Phase-1 length |
| `--epochs` | `50` | Total epochs |
| `--pretrained_ckpt` | `None` | 300W-LP checkpoint for KFace fine-tuning |

---

## Dataset Setup

**300W-LP** — pretraining

- Download: [3DDFA project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- Extract into `300W_LP/` (subdirs: `AFW/`, `HELEN/`, `IBUG/`, `LFPW/`, `*_Flip/`)
- Annotations: `.mat` files with rotation matrix → Euler angles

**AIHub KFace** — fine-tuning

- Source: [AIHub](https://aihub.or.kr) Dataset #83
- Pose encoded in filename: `{ID}_{acc}_{light}_{expr}_{C1–C20}.jpg`
- 20 camera positions covering ±90° yaw, −15° / +30° pitch

---

## Other Scripts

| Script | Usage |
|---|---|
| `predict.py` | Save annotated prediction grid from a dataset |
| `inference.py` | Real-time webcam demo (Haar cascade, press `q` to quit) |
| `evaluate.py` | Per-axis MAE + direction accuracy report |

---

## Project Structure

```
├── model.py          # EfficientNetHeadPose, HeadPoseLoss
├── dataset.py        # KFace parser and DataLoader
├── train.py          # KFace fine-tuning
├── train_300wlp.py   # 300W-LP pretraining
├── evaluate.py       # Evaluation
├── predict.py        # Batch visualization
├── inference.py      # Webcam inference
├── checkpoints/      # Model weights        (gitignored)
└── 300W_LP/          # Dataset              (gitignored)
```
