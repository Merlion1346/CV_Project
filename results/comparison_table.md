# Head Pose Comparison (AFLW2000, MAE in °)

| Model | Yaw MAE | Pitch MAE | Roll MAE | Mean MAE | FPS |
|-------|---------|-----------|----------|----------|-----|
| HopeNet | 6.47 | 6.56 | 5.44 | 6.16 | — |
| WHENet | — | — | — | — | — |
| TEST-Model B0 (300W-LP, 50 iter) | 13.7852 | 8.6076 | 8.8698 | 10.4209 | 47.5 |
| TEST-Model B1 (300W-LP, 50 iter) | 22.8804 | 9.5568 | 10.3881 | 14.2751 | 31.1 |
| ResNet50 (300W-LP, 50 iter) | 14.4037 | 7.8212 | 9.6905 | 10.6385 | 19.8 |
| MobileNetV2 (300W-LP, 50 iter) | 19.8396 | 9.8175 | 10.2313 | 13.2961 | 64.5 |

*Source: `results/experiment_1_baseline/metrics.csv` (FPS = CPU)*

---

## Experiment 2 — Ablation (EfficientNet-B0)

| Model | Yaw MAE | Pitch MAE | Roll MAE | Mean MAE |
|-------|---------|-----------|----------|----------|
| Exp1 Vanilla | 8.4007 | 6.4721 | 4.9743 | 6.6157 |
| Exp2 + FlipAug | 8.5247 | 6.8219 | 6.1512 | 7.1659 |
| Exp3 + RotAug | 9.6922 | 6.8081 | 7.6478 | 8.0494 |
| Exp4 + WeightedLoss | 6.7857 | 7.6162 | 8.4116 | 7.6045 |
| Exp5 + GeM | 6.7115 | 7.7423 | 8.0718 | 7.5085 |
| Exp6 All combined | 6.8967 | 7.6717 | 9.1958 | 7.9214 |

*Source: `results/experiment_2_ablation/ablation_results.csv`*

---

## Experiment 3 — Loss Comparison (EfficientNet-B0)

| Model | Yaw MAE | Pitch MAE | Roll MAE | Mean MAE |
|-------|---------|-----------|----------|----------|
| MSE only | 7.7631 | 6.6764 | 6.7526 | 7.064 |
| Combined Wrapped (WHENet) | 9.3216 | 7.082 | 4.8374 | 7.0803 |
| Combined CE+MSE (HopeNet) | 10.6199 | 7.2025 | 6.1633 | 7.9952 |
| Focal MSE | 10.6635 | 8.3684 | 8.8734 | 9.3018 |
| CE only | 26.5427 | 13.7859 | 13.3885 | 17.9057 |

*Source: `results/experiment_3_loss/loss_comparison.csv`*
