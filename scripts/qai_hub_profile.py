"""
Submit EfficientNetHeadPose to Qualcomm AI Hub for compile + profile.
Target: Snapdragon 8 Gen series  |  Runtime: ONNX (QNN Execution Provider)

Prerequisites:
    pip install qai-hub onnx
    qai-hub configure --api_token <YOUR_TOKEN>   # one-time setup

Usage:
    # 1. Export ONNX first
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx --verify

    # 2. Compile + profile on device
    python qai_hub_profile.py --onnx model.onnx

    # 3. Compile only (skip profiling)
    python qai_hub_profile.py --onnx model.onnx --compile_only

    # 4. Use a specific device (default: Snapdragon 8 Gen 2)
    python qai_hub_profile.py --onnx model.onnx --device "Snapdragon 8 Gen 3"
"""

import argparse
import sys

ANGLE_MAX = 99.0
DEFAULT_DEVICE = "Snapdragon 8 Gen 2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",         type=str, default="model.onnx",
                   help="Path to exported ONNX model")
    p.add_argument("--device",       type=str, default=DEFAULT_DEVICE,
                   help="AI Hub device name")
    p.add_argument("--img_size",     type=int, default=224)
    p.add_argument("--compile_only", action="store_true",
                   help="Skip profiling after compile")
    p.add_argument("--list_devices", action="store_true",
                   help="Print available devices and exit")
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


def compile_model(hub, onnx_path: str, device_name: str, img_size: int):
    print(f"[AI Hub] Submitting compile job...")
    print(f"         Model : {onnx_path}")
    print(f"         Device: {device_name}")

    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=hub.Device(device_name),
        input_specs={"image": ((1, 3, img_size, img_size), "float32")},
        options="--target_runtime onnx",
    )
    print(f"[AI Hub] Compile job ID: {compile_job.job_id}")
    print(f"         Waiting for compile to finish...")

    compile_job.wait()
    status = compile_job.get_status()
    print(f"[AI Hub] Compile status: {status.symbol}  {status.message}")

    if status.failure:
        print("[ERROR] Compile failed. Check AI Hub console for details.")
        sys.exit(1)

    return compile_job.get_target_model()


def profile_model(hub, target_model, device_name: str):
    print(f"\n[AI Hub] Submitting profile job on {device_name}...")

    profile_job = hub.submit_profile_job(
        model=target_model,
        device=hub.Device(device_name),
    )
    print(f"[AI Hub] Profile job ID: {profile_job.job_id}")
    print(f"         Waiting for profiling to finish...")

    profile_job.wait()
    status = profile_job.get_status()
    print(f"[AI Hub] Profile status: {status.symbol}  {status.message}")

    if status.failure:
        print("[ERROR] Profiling failed.")
        return

    profile = profile_job.download_profile()
    print_profile_summary(profile, device_name)


def print_profile_summary(profile, device_name: str):
    # profile is a list of layer dicts from download_profile()
    layers = profile if isinstance(profile, list) else []

    print(f"\n{'='*55}")
    print(f"  Profile Results — {device_name}")
    print(f"{'='*55}")

    if layers:
        total_us = sum(
            l.get("execution_time", l.get("execution_time_microseconds", 0))
            for l in layers
        )
        total_ms = total_us / 1000

        # compute unit breakdown
        unit_times: dict[str, float] = {}
        for l in layers:
            unit = l.get("compute_unit", l.get("delegate", "unknown"))
            t    = l.get("execution_time", l.get("execution_time_microseconds", 0))
            unit_times[unit] = unit_times.get(unit, 0) + t

        print(f"  Total inference time : {total_ms:.2f} ms")
        print(f"  Throughput           : {1000/total_ms:.1f} FPS  (single-image)")
        print(f"  Total layers         : {len(layers)}")
        print(f"\n  Compute unit breakdown:")
        for unit, us in sorted(unit_times.items(), key=lambda x: -x[1]):
            print(f"    {unit:<12} {us/1000:.2f} ms  ({100*us/total_us:.1f}%)")

        print(f"\n  Top-5 slowest layers:")
        key = lambda x: x.get("execution_time", x.get("execution_time_microseconds", 0))
        for i, layer in enumerate(sorted(layers, key=key, reverse=True)[:5], 1):
            t    = key(layer) / 1000
            name = layer.get("name", layer.get("op_type", "?"))
            print(f"    {i}. {name:<40} {t:.3f} ms")
    else:
        print("  (no layer detail available — check raw profile above)")

    print(f"{'='*55}\n")
    print("  Post-processing reminder:")
    print(f"    pred_degrees = model_output × {ANGLE_MAX}")
    print(f"    → [yaw°, pitch°, roll°]\n")


def main():
    args = parse_args()
    hub  = import_qai_hub()

    if args.list_devices:
        list_devices(hub)
        return

    target_model = compile_model(hub, args.onnx, args.device, args.img_size)

    if not args.compile_only:
        profile_model(hub, target_model, args.device)
    else:
        print("[AI Hub] Compile-only mode — skipping profile.")
        print(f"         Compiled model saved in AI Hub. Job ID logged above.")


if __name__ == "__main__":
    main()
