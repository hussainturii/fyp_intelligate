"""
benchmark_face_pipeline.py — Active Face Perception Pipeline Benchmark
=====================================================================
Milestone 3C: Benchmarks the active runtime path:
    Input Frame → YuNet (Face Detection) → MiniFASNetV2 (PAD) → SFace (Recognition)
    → Event Dispatch → SoftwareRuntime Access Policy

Measures:
- Stage-by-stage component latency (YuNet, MiniFASNetV2, SFace embedding & match)
- Full-pipeline end-to-end latency (p50, p90, p95, mean, min, max)
- Sustained frames per second (FPS)
- CPU vs. GPU (RTX 4060 Ti) comparison using identical real face test frames
- Outcome distribution (Bona Fide, Suspected Attack, Inconclusive, Not Evaluated)

Machine-readable results written to logs/benchmarks/m03c_face_pipeline_performance.json
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module1_face_recognition.anti_spoofing.deep_pad import MiniFASNetV2Detector
from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline

OUTPUT_JSON = ROOT / "logs" / "benchmarks" / "m03c_face_pipeline_performance.json"


def collect_test_images(num_samples: int = 100) -> List[Path]:
    """Collect test images from local LFW dataset."""
    lfw_dir = ROOT / "data" / "lfw" / "lfw"
    if not lfw_dir.exists():
        lfw_dir = ROOT / "data" / "lfw"
    all_images = sorted(list(lfw_dir.rglob("*.jpg")))
    if not all_images:
        raise FileNotFoundError(f"No face images found under {lfw_dir}")
    # Sample every k-th image to get wide identity diversity
    step = max(1, len(all_images) // num_samples)
    sampled = all_images[::step][:num_samples]
    return sampled


def run_benchmark_for_device(
    device_name: str,
    image_paths: List[Path],
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """Run full pipeline benchmark on the given device."""
    logger.info(f"[Benchmark] Setting up test pipeline on device: {device_name}")
    
    # In-memory store and runtime
    db_path = ROOT / f"benchmark_temp_{device_name.replace(':', '_')}.db"
    if db_path.exists():
        db_path.unlink()

    store = EventStore(str(db_path))
    runtime = SoftwareRuntime(store)
    face_engine = FaceEngine(store)

    # Register 3 known reference gallery identities
    for i, img_path in enumerate(image_paths[:3]):
        reg_img = cv2.imread(str(img_path))
        try:
            face_engine.register(f"Registered_User_{i+1}", "employee", reg_img)
            logger.info(f"[Benchmark] Enrolled Registered_User_{i+1}")
        except Exception as exc:
            logger.warning(f"[Benchmark] Enrolment skipped for {img_path.name}: {exc}")

    events_emitted = []
    def on_event(evt):
        events_emitted.append(evt)
        runtime.observe(evt, trusted=True)

    pipeline = ReplayPipeline(
        mode="face",
        runtime=runtime,
        callback=on_event,
        face_engine=face_engine,
        device=device_name,
    )

    # Preload and resize frames to 640x480 standard CCTV resolution
    frames = []
    for p in image_paths:
        raw = cv2.imread(str(p))
        if raw is not None:
            # Place 250x250 face into a 640x480 realistic gate camera frame
            canvas = np.full((480, 640, 3), 110, dtype=np.uint8)
            # Center the face
            y_off = (480 - raw.shape[0]) // 2
            x_off = (640 - raw.shape[1]) // 2
            canvas[y_off : y_off + raw.shape[0], x_off : x_off + raw.shape[1]] = raw
            frames.append(canvas)

    logger.info(f"[Benchmark] Loaded {len(frames)} test frames at 640x480.")

    # Warmup runs
    for i in range(min(warmup_runs, len(frames))):
        pipeline.process_frame(frames[i])

    # Timed benchmark runs
    latencies = []
    stage_latencies = {
        "detection_ms": [],
        "pad_ms": [],
        "recognition_ms": [],
        "tracking_ms": [],
    }

    pad_outcomes = {
        "BONA_FIDE_CANDIDATE": 0,
        "ATTACK_SUSPECTED": 0,
        "INCONCLUSIVE": 0,
        "NOT_EVALUATED": 0,
    }

    total_faces_detected = 0

    for idx, frame in enumerate(frames):
        t_start = time.perf_counter()
        res = pipeline.process_frame(frame)
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        latencies.append(elapsed_ms)

        stages = pipeline.metrics.get("stage_latencies", {})
        for k in stage_latencies:
            if k in stages:
                stage_latencies[k].append(stages[k])

        # Track PAD statuses
        face_results = pipeline.metrics.get("face_results", [])
        total_faces_detected += len(face_results)
        for f in face_results:
            st = f.get("pad_status", "NOT_EVALUATED")
            pad_outcomes[st] = pad_outcomes.get(st, 0) + 1

    # Cleanup temp db
    store.close()
    for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    arr = np.array(latencies)
    results = {
        "device": device_name,
        "num_frames": len(frames),
        "total_faces_processed": total_faces_detected,
        "pad_outcomes": pad_outcomes,
        "pipeline_latency_ms": {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "median_p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        },
        "sustained_fps": round(1000.0 / float(np.mean(arr)), 2),
        "stage_latencies_ms": {
            k: {
                "mean": round(float(np.mean(v)), 2) if v else 0.0,
                "median_p50": round(float(np.percentile(v, 50)), 2) if v else 0.0,
                "p95": round(float(np.percentile(v, 95)), 2) if v else 0.0,
            }
            for k, v in stage_latencies.items()
        },
    }
    return results


def main():
    logger.info("=================================================================")
    logger.info("INTELLIGATE — MILESTONE 3C FACE PIPELINE BENCHMARK")
    logger.info("=================================================================")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    test_images = collect_test_images(num_samples=100)
    logger.info(f"Sampled {len(test_images)} bona fide test images from local LFW.")

    # 1. CPU Evaluation
    cpu_results = run_benchmark_for_device("cpu", test_images, warmup_runs=5)

    # 2. GPU Evaluation (if available)
    cuda_available = torch.cuda.is_available()
    gpu_results = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA GPU detected: {gpu_name}")
        gpu_results = run_benchmark_for_device("cuda:0", test_images, warmup_runs=5)

    # Summary report
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if cuda_available else "N/A",
            "cuda_available": cuda_available,
            "torch_version": torch.__version__,
        },
        "models": {
            "face_detector": "YuNet (ONNX)",
            "pad_model": "MiniFASNetV2-80x80-2.7 (PyTorch)",
            "face_recognizer": "SFace (ONNX)",
        },
        "dataset": {
            "source": "LFW (Labeled Faces in the Wild) local dataset",
            "resolution": "640x480 standard canvas",
            "samples": len(test_images),
        },
        "cpu_benchmark": cpu_results,
        "gpu_benchmark": gpu_results,
    }

    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info(f"Wrote benchmark results to {OUTPUT_JSON}")

    # Print comparative table
    print("\n" + "=" * 75)
    print("INTELLIGATE — FACE PERCEPTION PIPELINE PERFORMANCE REPORT (M03C)")
    print("=" * 75)
    print(f"Test Samples: {len(test_images)} frames (640x480) | Hardware: {all_results['hardware']['gpu']}")
    print("-" * 75)
    print(f"{'Metric':<30} | {'CPU Fallback':<18} | {'CUDA:0 (RTX 4060 Ti)':<18}")
    print("-" * 75)
    
    cpu_pipe = cpu_results["pipeline_latency_ms"]
    gpu_pipe = gpu_results["pipeline_latency_ms"] if gpu_results else {}
    
    print(f"{'Pipeline Mean Latency':<30} | {cpu_pipe['mean']:>14.2f} ms | {gpu_pipe.get('mean', 0.0):>14.2f} ms")
    print(f"{'Pipeline p50 (Median)':<30} | {cpu_pipe['median_p50']:>14.2f} ms | {gpu_pipe.get('median_p50', 0.0):>14.2f} ms")
    print(f"{'Pipeline p95 Latency':<30} | {cpu_pipe['p95']:>14.2f} ms | {gpu_pipe.get('p95', 0.0):>14.2f} ms")
    print(f"{'Sustained Pipeline FPS':<30} | {cpu_results['sustained_fps']:>14.2f} FPS| {gpu_results['sustained_fps'] if gpu_results else 0.0:>14.2f} FPS")
    print("-" * 75)
    print("STAGE BREAKDOWN (Mean Latency):")
    for stage in ["detection_ms", "pad_ms", "recognition_ms", "tracking_ms"]:
        cpu_st = cpu_results["stage_latencies_ms"].get(stage, {}).get("mean", 0.0)
        gpu_st = gpu_results["stage_latencies_ms"].get(stage, {}).get("mean", 0.0) if gpu_results else 0.0
        print(f"  - {stage:<26} | {cpu_st:>14.2f} ms | {gpu_st:>14.2f} ms")
    print("-" * 75)
    print("PAD OUTCOMES (Bona Fide Test Set):")
    for st, count in cpu_results["pad_outcomes"].items():
        print(f"  - {st:<26} | {count:>14}    | {gpu_results['pad_outcomes'].get(st, 0) if gpu_results else 0:>14}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
