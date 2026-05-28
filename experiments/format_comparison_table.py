#!/usr/bin/env python3
"""
실험 결과를 논문/보고용 비교 표로 출력합니다.

Usage:
  python head_pose_experiments/format_comparison_table.py
  python head_pose_experiments/format_comparison_table.py \\
      --csv results/experiment_1_baseline/metrics.csv \\
      --epochs 50 --output results/comparison_table.md
"""

import argparse
from pathlib import Path

import pandas as pd


# ── 논문 / 외부 baseline (수동 입력) ─────────────────────
LITERATURE_BASELINES = [
    {
        "model": "HopeNet",
        "yaw_mae": 6.47,
        "pitch_mae": 6.56,
        "roll_mae": 5.44,
        "mean_mae": 6.16,
        "fps": None,
    },
    {
        "model": "WHENet",
        "yaw_mae": None,
        "pitch_mae": None,
        "roll_mae": None,
        "mean_mae": None,
        "fps": None,
    },
]

# metrics.csv model 컬럼 → 표시 이름
MODEL_DISPLAY_NAMES = {
    "resnet50":        "ResNet50 (300W-LP, {epochs} iter)",
    "mobilenetv2":     "MobileNetV2 (300W-LP, {epochs} iter)",
    "efficientnet_b0": "TEST-Model B0 (300W-LP, {epochs} iter)",
    "efficientnet_b1": "TEST-Model B1 (300W-LP, {epochs} iter)",
}

# 추가 실험 결과 (다른 스크립트 / 수동 실험)
EXTRA_ROWS = [
    # 예시 — 실험 후 숫자 채우기
    # {
    #     "model": "TEST-Model B0 (300W-LP, 50 iter + soft argmax)",
    #     "yaw_mae": 4.4566,
    #     "pitch_mae": 9.3258,
    #     "roll_mae": 8.9426,
    #     "mean_mae": 7.5750,
    #     "fps": None,
    # },
    # {
    #     "model": "TEST-Model B0 (300W-LP + k_face, 50 iter)",
    #     "yaw_mae": 30.7665,
    #     "pitch_mae": 17.2528,
    #     "roll_mae": 16.3134,
    #     "mean_mae": 21.4442,
    #     "fps": None,
    # },
]


def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    return str(val)


def csv_to_rows(csv_path: Path, epochs: int, fps_col: str = "fps_cpu") -> list[dict]:
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        key = r["model"]
        name = MODEL_DISPLAY_NAMES.get(key, f"{key} (300W-LP, {epochs} iter)")
        name = name.format(epochs=epochs)
        fps = r[fps_col] if fps_col in df.columns else None
        rows.append({
            "model":     name,
            "yaw_mae":   r["yaw_mae"],
            "pitch_mae": r["pitch_mae"],
            "roll_mae":  r["roll_mae"],
            "mean_mae":  r["mean_mae"],
            "fps":       fps,
        })
    return rows


def build_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae", "fps",
    ]).rename(columns={
        "model":     "Model",
        "yaw_mae":   "Yaw MAE",
        "pitch_mae": "Pitch MAE",
        "roll_mae":  "Roll MAE",
        "mean_mae":  "Mean MAE",
        "fps":       "FPS",
    })


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Model | Yaw MAE | Pitch MAE | Roll MAE | Mean MAE | FPS |",
        "|-------|---------|-----------|----------|----------|-----|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} "
            f"| {fmt(r['Yaw MAE'])} "
            f"| {fmt(r['Pitch MAE'])} "
            f"| {fmt(r['Roll MAE'])} "
            f"| {fmt(r['Mean MAE'])} "
            f"| {fmt(r['FPS'], decimals=1)} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Format head pose results as comparison table")
    p.add_argument("--csv",       default="results/experiment_1_baseline/metrics.csv")
    p.add_argument("--epochs",    type=int, default=50,
                   help="학습 epoch 수 (표시 이름에 사용)")
    p.add_argument("--fps_col",   default="fps_cpu",
                   choices=["fps_cpu", "fps_gpu"],
                   help="FPS 컬럼 선택")
    p.add_argument("--output",    default="results/comparison_table.md")
    p.add_argument("--no_literature", action="store_true",
                   help="HopeNet/WHENet baseline 행 제외")
    args = p.parse_args()

    rows = []
    if not args.no_literature:
        rows.extend(LITERATURE_BASELINES)

    csv_path = Path(args.csv)
    if csv_path.exists():
        rows.extend(csv_to_rows(csv_path, args.epochs, args.fps_col))
    else:
        print(f"[Warn] CSV not found: {csv_path}")

    rows.extend(EXTRA_ROWS)

    df = build_table(rows)

    print("\n" + "=" * 72)
    print("  HEAD POSE COMPARISON (AFLW2000, MAE in degrees)")
    print("=" * 72)
    print(df.to_string(index=False))
    print()

    md = to_markdown(df)
    print(md)
    print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")

    csv_out = out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {out}")
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
