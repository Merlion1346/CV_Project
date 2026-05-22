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

# 2. Evaluate on AFLW2000-3D
python evaluate_aflw2000.py --checkpoint checkpoints/best.pth --data_dir AFLW2000/

# 3. Evaluate on KFace
python evaluate.py --checkpoint checkpoints/best.pth --data_dir /path/to/kface_data

# 4. Live webcam demo
python inference.py --checkpoint checkpoints/best.pth
```

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNet B0–B7 (ImageNet pretrained) |
| Pooling | GeMPooling (replaces AdaptiveAvgPool2d) |
| Attention | Channel Attention (lightweight CBAM) |
| Head | Linear 512 → 128 → 3 · BN · SiLU · Dropout |
| Output | Normalized [−1, 1] → ×99° = degrees |
| Loss | Huber (delta=1.0, equal axis weights) |

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
| Horizontal flip | Yaw and roll signs flipped to maintain label consistency |
| Random rotation | ±15° with roll label correction |
| Color jitter | Applied during training |

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

### Training Outputs

Each run writes to `--output_dir` (default: `./checkpoints/`):

| File | Description |
|---|---|
| `best.pth` | Best validation-loss checkpoint |
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
├── model.py                  # EfficientNetHeadPose, GeMPooling, HeadPoseLoss
├── dataset.py                # KFace parser and DataLoader
├── train.py                  # 300W-LP training
├── evaluate.py               # KFace evaluation
├── evaluate_aflw2000.py      # AFLW2000-3D evaluation (HopeNet protocol)
├── predict.py                # Batch visualization
├── inference.py              # Webcam inference
├── export_onnx.py            # ONNX export
├── qai_hub_profile.py        # QAI Hub profiling
├── checkpoints/              # Model weights + train_log.csv + training_plot.png  (gitignored)
└── 300W_LP/                  # Dataset                                            (gitignored)
```
