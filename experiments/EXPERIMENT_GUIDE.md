# Head Pose Estimation 실험 가이드

## 🎯 실험 로드맵

```
Phase 1: 환경 준비       └─ 데이터셋 다운로드 및 전처리
Phase 2: Baseline 구축   └─ Experiment 1: Baseline 비교 (1-2일)
Phase 3: 핵심 개선       └─ Experiment 2: Ablation Study (3-5일)
Phase 4: 최적화          ├─ Experiment 3: Loss Function 비교
                         ├─ Experiment 4: Data Augmentation 영향
                         └─ Experiment 5: 입력 해상도 실험 (2-3일)
Phase 5: 최종 평가       └─ 벤치마크 테스트 및 분석 (1일)
```

---

## 🚀 빠른 시작

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비 (300W-LP, AFLW2000)
# data/300W_LP/, data/AFLW2000/ 에 배치

데이터 스캔
python head_pose_experiments/experiments/01_baseline_comparison.py \
  --data_dir 300W_LP/ --test_dir AFLW2000 --quick_test


# 3. 전체 실험 자동 실행
bash run_all_experiments.sh

# 개별 실험 실행
python run_experiments.py --exp 1    # Baseline 비교만
python run_experiments.py --exp 2    # Ablation Study만
python run_experiments.py --exp all  # 전체 실험
```

---

## 📊 실험별 상세 가이드

### Experiment 1: Baseline 비교 (예상 소요: 8-12시간)

```bash
python experiments/01_baseline_comparison.py \
    --data_dir data/300W_LP \
    --test_dir data/AFLW2000 \
    --epochs 50 \
    --batch_size 512
```

**목적**: 백본 네트워크 성능 비교 (모두 ImageNet Pretrained)

| 모델 | 파라미터 | 비고 |
|------|---------|------|
| ResNet50 | 25.6M | HopeNet 기준 모델 |
| MobileNetV2 | 3.5M | 경량 모델 |
| EfficientNet-B0 | 5.3M | **제안 모델** |
| EfficientNet-B1 | 7.8M | B0 확장 버전 |

**출력**:
- `results/experiment_1_baseline/metrics.csv`
- `results/experiment_1_baseline/comparison_plot.png`

---

### Experiment 2: Ablation Study (예상 소요: 12-16시간)

```bash
python experiments/02_ablation_study.py \
    --data_dir data/300W_LP \
    --test_dir data/AFLW2000 \
    --epochs 50 \
    --batch_size 512
```

**목적**: 각 구성 요소의 기여도 분석

| 실험 | 구성 | 목적 |
|------|------|------|
| Exp 1 | Vanilla EfficientNet-B0 | 베이스라인 |
| Exp 2 | + Flip augmentation | yaw/roll 부호 반전 |
| Exp 3 | + Random rotation | roll 보정 |
| Exp 4 | + Weighted loss | pitch/roll 가중치 |
| Exp 5 | + GeM pooling | 풀링 개선 |
| Exp 6 | All combined | 최종 모델 |

**출력**:
- `results/experiment_2_ablation/ablation_results.csv`
- `results/experiment_2_ablation/ablation_plot.png`

---

### Experiment 3: Loss Function 비교 (예상 소요: 10-14시간)

```bash
python experiments/03_loss_comparison.py \
    --data_dir data/300W_LP \
    --test_dir data/AFLW2000 \
    --epochs 50
```

**목적**: 최적의 손실 함수 조합 찾기

| Loss | 설명 |
|------|------|
| MSE only | 기본 회귀 |
| CE only | 분류 기반 |
| Combined (CE+MSE) | HopeNet 방식 |
| Combined + Wrapped | WHENet 방식 |
| Focal MSE | 큰 오차 집중 |

---

### Experiment 4: Data Augmentation 영향 (예상 소요: 8-12시간)

```bash
python experiments/04_augmentation_study.py \
    --data_dir data/300W_LP \
    --test_dir data/AFLW2000 \
    --epochs 50
```

**목적**: 데이터 증강 강도별 성능 분석

| 레벨 | 구성 |
|------|------|
| None | 증강 없음 |
| Basic | Flip only |
| Medium | Flip + Rotation + Color |
| Heavy | Flip + Rotation + Color + Blur |

---

### Experiment 5: 입력 해상도 실험 (예상 소요: 6-10시간)

```bash
python experiments/05_resolution_study.py \
    --data_dir data/300W_LP \
    --test_dir data/AFLW2000 \
    --epochs 30
```

**목적**: 입력 해상도에 따른 정확도-속도 trade-off

**해상도**: 128×128 / 160×160 / 192×192 / 224×224 / 256×256

---

## 📏 평가 메트릭

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| MAE Yaw | 좌우 회전 오차 | ≤ 5° |
| MAE Pitch | 상하 회전 오차 | ≤ 5° |
| MAE Roll | 기울기 오차 | ≤ 5° |
| Mean MAE | 3축 평균 | ≤ 5° |
| FPS (CPU) | 초당 프레임 | ≥ 20 |

---

## 🎓 예상 결과

```
=== Baseline Comparison (AFLW2000) ===
ResNet50:        Mean MAE = 6.16° (25.6M params, 15 FPS)
MobileNetV2:     Mean MAE = 7.34° ( 3.5M params, 32 FPS)
EfficientNet-B0: Mean MAE = 5.89° ( 5.3M params, 28 FPS) ← 제안
EfficientNet-B1: Mean MAE = 5.67° ( 7.8M params, 24 FPS)

=== Ablation Study ===
Vanilla:         Mean MAE = 7.93°
+ Flip aug:      Mean MAE = 6.64° (Δ -1.29°)
+ Rotation:      Mean MAE = 6.19° (Δ -0.45°)
+ Weighted loss: Mean MAE = 5.60° (Δ -0.59°)
+ GeM pooling:   Mean MAE = 5.22° (Δ -0.38°)
All combined:    Mean MAE = 4.89° ✓ 목표 달성
```

---

## 📝 실험 체크리스트

- [ ] Phase 1: 데이터셋 준비 (300W-LP, AFLW2000)
- [ ] Phase 2: Baseline 실험 완료
- [ ] Phase 3: Ablation Study 완료
- [ ] Phase 4: Loss / Augmentation / Resolution 실험 완료
- [ ] Phase 5: 최종 벤치마크 테스트
- [ ] Phase 6: 결과 보고서 작성

---

## 🐛 문제 해결

```bash
# CUDA Out of Memory
python experiments/01_baseline_comparison.py --batch_size 256

# 실험 중단 후 재개
python experiments/01_baseline_comparison.py --resume

# 빠른 테스트 (5 epoch)
python run_experiments.py --quick_test --epochs 5
```

---

## 📚 참고

- HopeNet: Fine-Grained Head Pose Estimation Without Keypoints (CVPR 2018)
- WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose (2020)
- 300W-LP: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
- AFLW2000: https://www.kaggle.com/datasets/mohamedadlyi/aflw2000-3d
