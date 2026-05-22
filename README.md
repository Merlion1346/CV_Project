# Head Pose Estimation

EfficientNet-based head pose estimation that predicts **yaw / pitch / roll** from a single face image.  
Pretrained on 300W-LP, evaluated on AFLW2000-3D.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 1. Train on 300W-LP
python train.py --data_dir /path/to/300W_LP --epochs 50

# 2. Resume interrupted training
python train.py --data_dir /path/to/300W_LP --resume

# 3. Evaluate on AFLW2000-3D
python evaluate_aflw2000.py --checkpoint checkpoints/best.pth --data_dir AFLW2000/

# 4. Evaluate on KFace
python evaluate.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data

# 5. Live webcam demo
python inference.py --checkpoint checkpoints/best.pth
```

---

## Benchmark

Evaluated on AFLW2000-3D (2000 samples), HopeNet protocol:

| Axis | MAE |
|---|---|
| Yaw | 5.13° |
| Pitch | 9.63° |
| Roll | 9.74° |
| **Mean** | **8.16°** |

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNet B0–B7 (ImageNet pretrained) |
| Pooling | AdaptiveAvgPool2d |
| Attention | Channel Attention (lightweight CBAM) |
| Head | Linear 512 → 128 → 3 · BN · SiLU · Dropout |
| Output | Normalized [−1, 1] → ×99° = degrees |
| Loss | Huber loss (smooth L1, delta=0.1) |

---

## Training

Two-phase schedule applies automatically:

| Phase | Condition | What trains |
|---|---|---|
| 1 — Warm-up | epochs 1 → `warmup_epochs` | Head + attention only |
| 2 — Fine-tune | `warmup_epochs + 1` → end | + top 3 backbone blocks (LR ×0.05) |

### Data Augmentation

| Transform | Detail |
|---|---|
| Random crop | Resize to img_size+32, then crop to img_size |
| Color jitter | brightness/contrast ±0.3, saturation ±0.2 |
| Random grayscale | p=0.05 |

### `train.py` Hyperparameters

| Argument | Default | Notes |
|---|---|---|
| `--variant` | `b0` | b0–b7; larger = better accuracy, more VRAM |
| `--img_size` | `224` | Recommended: 380 for b4+, 456 for b5+ |
| `--batch_size` | `64` | Reduce to 32 for b5+ on 8 GB VRAM |
| `--lr` | `3e-4` | AdamW base LR |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--dropout` | `0.3` | Head dropout rate |
| `--warmup_epochs` | `5` | Phase-1 length |
| `--epochs` | `50` | Total epochs |
| `--val_ratio` | `0.1` | Validation split |
| `--num_workers` | `4` | DataLoader workers |
| `--resume` | `False` | Resume training from `last.pth` |

### Training Outputs

Each run writes to `--output_dir` (default: `./checkpoints/`):

| File | Description |
|---|---|
| `best.pth` | Best checkpoint by validation MAE |
| `last.pth` | Latest checkpoint (used by `--resume`) |
| `train_log.csv` | Per-epoch metrics (loss, MAE per axis) |
| `training_plot.png` | Loss and MAE curves (updated every epoch) |
| `tb/` | TensorBoard event files |

```bash
tensorboard --logdir checkpoints/tb --port 30011
# → http://localhost:30011
```

---

## Dataset Setup

**300W-LP** — training

- Download: [3DDFA project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- Extract into `300W_LP/` (subdirs: `AFW/`, `HELEN/`, `IBUG/`, `LFPW/`, `*_Flip/`)
- Annotations: `.mat` files with rotation matrix → Euler angles

**AIHub KFace** — evaluation

- Source: [AIHub](https://aihub.or.kr) Dataset #83
- Pose encoded in filename: `{ID}_{acc}_{light}_{expr}_{C1–C20}.jpg`
- 20 camera positions covering ±90° yaw, −15° / +30° pitch

**AFLW2000-3D** — evaluation

- Download: [3DDFA project page](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- Fitted 3D faces of the first 2000 AFLW samples; standard benchmark for 3D face alignment and head pose evaluation
- Annotations: `.mat` files with fitted 3DMM parameters → Euler angles (yaw, pitch, roll)

---

## Other Scripts

**Evaluate on KFace**
```bash
python evaluate.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data
python evaluate.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data --save  # export JSON
```
Reports per-axis MAE (yaw / pitch / roll) and direction accuracy (front / left / right / up / down).

**Evaluate on AFLW2000-3D** (HopeNet baseline protocol)
```bash
python evaluate_aflw2000.py --checkpoint checkpoints/best.pth --data_dir AFLW2000/
```
Reports MAE in degrees for yaw / pitch / roll following the exact protocol from [HopeNet](https://github.com/natanielruiz/deep-head-pose/blob/master/code/test_hopenet.py): face is cropped from pt2d landmarks (20 % margin), resized to 224 × 224, and per-axis MAE is averaged over all 2000 samples.

**Prediction grid**
```bash
python predict.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data \
  --num_samples 100 --output_dir predictions/
```

**Live webcam**
```bash
python inference.py --checkpoint checkpoints/best.pth  # press q to quit
```

---

## Project Structure

```
├── model.py                  # EfficientNetHeadPose, ChannelAttention, HeadPoseLoss
├── dataset.py                # KFace parser and DataLoader
├── train.py                  # 300W-LP training (TensorBoard, CSV log, resume)
├── evaluate.py               # KFace evaluation
├── evaluate_aflw2000.py      # AFLW2000-3D evaluation (HopeNet protocol)
├── predict.py                # Batch visualization
├── inference.py              # Webcam inference
├── export_onnx.py            # ONNX export
├── qai_hub_profile.py        # QAI Hub profiling
├── checkpoints/              # best.pth, last.pth, train_log.csv, training_plot.png, tb/
└── 300W_LP/                  # Dataset                                            (gitignored)
```
