#!/bin/bash
# run_all_experiments.sh
# 전체 실험 순서대로 자동 실행

set -e

DATA_DIR="data/300W_LP"
TEST_DIR="data/AFLW2000"
EPOCHS=50
BATCH=512

echo "================================================"
echo "  Head Pose Estimation - Full Experiment Pipeline"
echo "================================================"
echo ""

# Exp 1: Baseline
echo "[1/5] Baseline Comparison..."
python experiments/01_baseline_comparison.py \
    --data_dir $DATA_DIR --test_dir $TEST_DIR \
    --epochs $EPOCHS --batch_size $BATCH

# Exp 2: Ablation Study
echo "[2/5] Ablation Study..."
python experiments/02_ablation_study.py \
    --data_dir $DATA_DIR --test_dir $TEST_DIR \
    --epochs $EPOCHS --batch_size $BATCH

# Exp 3: Loss Function
echo "[3/5] Loss Function Comparison..."
python experiments/03_loss_comparison.py \
    --data_dir $DATA_DIR --test_dir $TEST_DIR \
    --epochs $EPOCHS --batch_size $BATCH

# Exp 4: Augmentation
echo "[4/5] Augmentation Study..."
python experiments/04_augmentation_study.py \
    --data_dir $DATA_DIR --test_dir $TEST_DIR \
    --epochs $EPOCHS --batch_size $BATCH

# Exp 5: Resolution
echo "[5/5] Resolution Study..."
python experiments/05_resolution_study.py \
    --data_dir $DATA_DIR --test_dir $TEST_DIR \
    --epochs 30 --batch_size $BATCH

echo ""
echo "================================================"
echo "  All experiments done!"
echo "  Results: results/"
echo "================================================"
