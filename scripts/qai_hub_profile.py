"""
Submit EfficientNetHeadPose to Qualcomm AI Hub for compile + profile.
Target: Snapdragon 8 Gen series  |  Runtime: ONNX (QNN Execution Provider)

Prerequisites:
    pip install qai-hub onnx
    qai-hub configure --api_token <YOUR_TOKEN>   # one-time setup

Usage:
    # 1. Export ONNX first
    python scripts/export_onnx.py --checkpoint models/checkpoints/best.pth --output models/model.onnx

    # 2. Profile on 20 devices (default)
    python scripts/qai_hub_profile.py

    # 3. Profile on specific devices (comma-separated)
    python scripts/qai_hub_profile.py --devices "Snapdragon 8 Gen 2,Snapdragon 8 Gen 3"

    # 4. Compile only (skip profiling)
    python scripts/qai_hub_profile.py --compile_only

    # 5. List available devices
    python scripts/qai_hub_profile.py --list_devices
"""

import argparse
import sys

ANGLE_MAX = 99.0

LOW_END_DEVICES = [
    # ~1–5 TOPS NPU / 구형 mid-range — CA73급 타겟과 가장 유사한 성능 구간
    "Google Pixel 3a",           # Snapdragon 670  / Hexagon 685  ~1.5 TOPS
    "Google Pixel 3a XL",        # Snapdragon 670  / Hexagon 685  ~1.5 TOPS
    "Samsung Galaxy Tab A8 (2021)",  # Snapdragon 662 / Hexagon 686  ~2 TOPS
    "Samsung Galaxy A14 5G",     # Snapdragon 480+ / Hexagon 686  ~2 TOPS
    "Xiaomi Redmi Note 10 5G",   # Dimensity 700   / APU          ~2 TOPS
    "Google Pixel 3",            # Snapdragon 845  / Hexagon 685  ~2 TOPS
    "Google Pixel 3 XL",         # Snapdragon 845  / Hexagon 685  ~2 TOPS
    "Google Pixel 4a",           # Snapdragon 730G / Hexagon 688  ~4 TOPS
    "Google Pixel 5",            # Snapdragon 765G / Hexagon 698  ~5 TOPS
    "Google Pixel 5a 5G",        # Snapdragon 765G / Hexagon 698  ~5 TOPS
]

DEFAULT_DEVICES = [
    # Google Pixel
    "Google Pixel 3 (Family)",
    "Google Pixel 3",
    "Google Pixel 3a",
    "Google Pixel 3a XL",
    "Google Pixel 3 XL",
    "Google Pixel 4",
    "Google Pixel 4a",
    "Google Pixel 5 (Family)",
    "Google Pixel 5",
    "Google Pixel 5a 5G",
    "Google Pixel 6 (Family)",
    "Google Pixel 6",
    "Google Pixel 7 (Family)",
    "Google Pixel 7",
    "Google Pixel 7 Pro",
    "Google Pixel 8 (Family)",
    "Google Pixel 8",
    "Google Pixel 8 Pro",
    "Google Pixel 9 (Family)",
    "Google Pixel 9",
    "Google Pixel 9 Pro",
    "Google Pixel 9 Pro XL",
    "Google Pixel 10",
    "Google Pixel 10 Pro XL",
    # Samsung Galaxy S
    "Samsung Galaxy S21 (Family)",
    "Samsung Galaxy S21",
    "Samsung Galaxy S21 Ultra",
    "Samsung Galaxy S22 (Family)",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S23 (Family)",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24 (Family)",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24+",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S25 (Family)",
    "Samsung Galaxy S25",
    "Samsung Galaxy S25+",
    "Samsung Galaxy S25 Ultra",
    # Samsung Galaxy A / Note / Tab
    "Samsung Galaxy Note 20 (Intl)",
    "Samsung Galaxy A14 5G",
    "Samsung Galaxy A53 5G",
    "Samsung Galaxy A73 5G",
    "Samsung Galaxy Tab S7",
    "Samsung Galaxy Tab S8",
    "Samsung Galaxy Tab A8 (2021)",
    # Xiaomi
    "Xiaomi Redmi Note 10 5G",
    "Xiaomi 12 (Family)",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    # Snapdragon 레퍼런스 보드
    "Snapdragon 7 Gen 4 QRD",
    "Snapdragon 8 Elite QRD",
    "Snapdragon 8 Elite Gen 5 QRD",
    "Snapdragon X Elite CRD",
    "Snapdragon X Plus 8-Core CRD",
    "Snapdragon X2 Elite CRD",
    # 임베디드 / IoT / 차량용
    "QCS8275 (Proxy)",
    "QCS8450 (Proxy)",
    "QCS8550 (Proxy)",
    "XR2 Gen 2 (Proxy)",
    "SA7255P ADP",
    "SA8295P ADP",
    "SA8775P ADP",
    "Dragonwing Q-6690 MTP",
    "Dragonwing RB3 Gen 2 Vision Kit",
    "Dragonwing IQ-9075 EVK",
]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",         type=str, default="models/model.onnx",
                   help="Path to exported ONNX model")
    p.add_argument("--devices",      type=str, default="",
                   help="Comma-separated device names (default: 20 preset devices)")
    p.add_argument("--img_size",     type=int, default=300)
    p.add_argument("--compile_only", action="store_true",
                   help="Skip profiling after compile")
    p.add_argument("--list_devices", action="store_true",
                   help="Print available devices and exit")
    p.add_argument("--low_end", action="store_true",
                   help="Use LOW_END_DEVICES (~1-5 TOPS) instead of DEFAULT_DEVICES")
    return p.parse_args()


def import_qai_hub():
    try:
        import qai_hub
        return qai_hub
    except ImportError:
        print("[ERROR] qai-hub not installed.")
        print("        pip install qai-hub")
        print("        qai-hub configure --api_token <YOUR_TOKEN>")
        sys.exit(1)


def list_devices(hub):
    print("\nAvailable devices on Qualcomm AI Hub:")
    for d in hub.get_devices():
        print(f"  {d.name}")
    print()


# ─────────────────────────────────────────────
# Compile (per device, parallel submit)
# ─────────────────────────────────────────────
def submit_compile_jobs(hub, onnx_path: str, devices: list[str], img_size: int) -> dict:
    """모든 디바이스에 컴파일 job을 동시 제출. {device_name: job} 반환."""
    jobs = {}
    for device_name in devices:
        print(f"  [Submit Compile] {device_name}")
        try:
            job = hub.submit_compile_job(
                model=onnx_path,
                device=hub.Device(device_name),
                input_specs={"image": ((1, 3, img_size, img_size), "float32")},
                options="--target_runtime onnx",
            )
            jobs[device_name] = job
        except Exception as e:
            print(f"  [WARN] Submit failed for {device_name}: {e}")
    return jobs


def wait_compile_jobs(jobs: dict) -> dict:
    """컴파일 완료 대기. {device_name: target_model} 반환 (실패 디바이스 제외)."""
    target_models = {}
    for device_name, job in jobs.items():
        job.wait()
        status = job.get_status()
        if status.failure:
            print(f"  [FAIL] Compile — {device_name}: {status.message}")
        else:
            print(f"  [OK]   Compile — {device_name}")
            target_models[device_name] = job.get_target_model()
    return target_models


# ─────────────────────────────────────────────
# Profile (parallel submit)
# ─────────────────────────────────────────────
def submit_profile_jobs(hub, target_models: dict) -> dict:
    """컴파일된 모델들에 대해 프로파일 job 동시 제출."""
    jobs = {}
    for device_name, target_model in target_models.items():
        print(f"  [Submit Profile] {device_name}")
        try:
            job = hub.submit_profile_job(
                model=target_model,
                device=hub.Device(device_name),
            )
            jobs[device_name] = job
        except Exception as e:
            print(f"  [WARN] Submit failed for {device_name}: {e}")
    return jobs


def wait_profile_jobs(jobs: dict) -> dict:
    """프로파일 완료 대기. {device_name: profile} 반환."""
    results = {}
    for device_name, job in jobs.items():
        job.wait()
        status = job.get_status()
        if status.failure:
            print(f"  [FAIL] Profile — {device_name}: {status.message}")
        else:
            profile = job.download_profile()
            results[device_name] = profile
            print(f"  [OK]   Profile — {device_name}")
    return results


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
def extract_stats(profile) -> dict | None:
    # QAI Hub returns a dict; fall back to list for legacy compatibility
    if isinstance(profile, dict):
        summary = profile.get("execution_summary", {})
        detail  = profile.get("execution_detail", [])
        if isinstance(detail, list):
            layers = detail
        elif isinstance(detail, dict):
            layers = (detail.get("op_profiles")
                      or detail.get("layers", []))
        else:
            layers = profile.get("layers", [])

        total_us = (summary.get("inference_time")
                    or summary.get("estimated_inference_time")
                    or summary.get("total_inference_time_us", 0))

        if not total_us and layers:
            total_us = sum(
                l.get("execution_time", l.get("execution_time_microseconds", 0))
                for l in layers
            )
    elif isinstance(profile, list):
        layers   = profile
        total_us = sum(
            l.get("execution_time", l.get("execution_time_microseconds", 0))
            for l in layers
        )
    else:
        return None

    if not total_us:
        return None

    unit_times: dict[str, float] = {}
    for l in (layers or []):
        unit = l.get("compute_unit", l.get("delegate", "unknown"))
        t    = l.get("execution_time", l.get("execution_time_microseconds", 0))
        unit_times[unit] = unit_times.get(unit, 0) + t

    return {
        "total_ms":   total_us / 1000,
        "fps":        1000 / (total_us / 1000) if total_us > 0 else 0,
        "n_layers":   len(layers) if layers else 0,
        "unit_times": unit_times,
    }


def print_summary(results: dict):
    print(f"\n{'='*70}")
    print(f"  Multi-Device Profile Summary")
    print(f"{'='*70}")
    print(f"  {'Device':<35} {'Latency(ms)':>12} {'FPS':>8} {'Layers':>8}")
    print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*8}")

    rows = []
    for device_name, profile in results.items():
        stats = extract_stats(profile)
        if stats:
            rows.append((device_name, stats))

    # 지연 시간 오름차순 정렬
    rows.sort(key=lambda x: x[1]["total_ms"])

    for device_name, stats in rows:
        print(f"  {device_name:<35} {stats['total_ms']:>11.2f}ms {stats['fps']:>7.1f} {stats['n_layers']:>8}")

    if rows:
        best  = rows[0]
        worst = rows[-1]
        print(f"\n  Fastest : {best[0]}  ({best[1]['total_ms']:.2f} ms)")
        print(f"  Slowest : {worst[0]}  ({worst[1]['total_ms']:.2f} ms)")

        print(f"\n  Compute unit breakdown (fastest device — {best[0]}):")
        unit_times = best[1]["unit_times"]
        total_us   = sum(unit_times.values())
        for unit, us in sorted(unit_times.items(), key=lambda x: -x[1]):
            print(f"    {unit:<14} {us/1000:.2f} ms  ({100*us/total_us:.1f}%)")

    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args    = parse_args()
    hub     = import_qai_hub()

    if args.list_devices:
        list_devices(hub)
        return

    devices = (
        [d.strip() for d in args.devices.split(",") if d.strip()]
        if args.devices else (LOW_END_DEVICES if args.low_end else DEFAULT_DEVICES)
    )
    print(f"[AI Hub] Target: {len(devices)} devices | Model: {args.onnx}")

    # ── Compile ──────────────────────────────
    print(f"\n[Phase 1] Submitting compile jobs...")
    compile_jobs   = submit_compile_jobs(hub, args.onnx, devices, args.img_size)
    print(f"\n[Phase 1] Waiting for {len(compile_jobs)} compile jobs...")
    target_models  = wait_compile_jobs(compile_jobs)
    print(f"[Phase 1] Done — {len(target_models)}/{len(devices)} compiled successfully")

    if args.compile_only:
        print("[AI Hub] Compile-only mode — skipping profile.")
        return

    # ── Profile ──────────────────────────────
    print(f"\n[Phase 2] Submitting profile jobs...")
    profile_jobs   = submit_profile_jobs(hub, target_models)
    print(f"\n[Phase 2] Waiting for {len(profile_jobs)} profile jobs...")
    profile_results = wait_profile_jobs(profile_jobs)
    print(f"[Phase 2] Done — {len(profile_results)}/{len(target_models)} profiled successfully")

    print_summary(profile_results)


if __name__ == "__main__":
    main()
