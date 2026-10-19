"""Measure real GPU (CUDA) inference and end-to-end pipeline latency on RTX 4060 Ti.
Compares directly against logs/benchmarks/cpu_baseline.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline, enhance
from module2_vehicle.plate_ocr import PlateOCR


def run_gpu_benchmark(video_path: str, num_frames: int = 16, sustained_frames: int = 300):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cuda_available = torch.cuda.is_available()
    device = "cuda:0" if cuda_available else "cpu"

    print(f"=== INTELLIGATE GPU BENCHMARK SUITE ===")
    print(f"CUDA Available: {cuda_available} | Device: {device_name}")
    print(f"PyTorch: {torch.__version__} | OpenCV: {cv2.__version__}")
    print(f"Platform: {platform.platform()}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(0, total_video_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (640, 360)))
    cap.release()

    if not frames:
        raise SystemExit("No frames extracted from video.")

    results = {}

    # 1. Direct YOLOv8n Inference: GPU vs CPU on the sampled video frames
    print("\n--- 1. YOLOv8n Detector Benchmark (Direct Inference) ---")
    yolo_model = YOLO(str(ROOT / "yolov8n.pt"))
    # Warmup GPU
    for _ in range(15):
        yolo_model.predict(frames[0], imgsz=416, conf=0.4, classes=[0, 2, 3, 5, 7], device=device, verbose=False)

    gpu_yolo_times = []
    for frame in frames:
        t0 = time.perf_counter()
        pred = yolo_model.predict(frame, imgsz=416, conf=0.4, classes=[0, 2, 3, 5, 7], device=device, verbose=False)
        gpu_yolo_times.append((time.perf_counter() - t0) * 1000)

    results["yolov8n_detector_gpu"] = {
        "device": device_name,
        "median_ms": round(statistics.median(gpu_yolo_times), 2),
        "mean_ms": round(statistics.mean(gpu_yolo_times), 2),
        "p95_ms": round(float(np.percentile(gpu_yolo_times, 95)), 2),
        "mean_inference_fps": round(1000.0 / statistics.mean(gpu_yolo_times), 2),
    }
    print(f"YOLOv8n GPU ({device}): median = {results['yolov8n_detector_gpu']['median_ms']} ms, "
          f"P95 = {results['yolov8n_detector_gpu']['p95_ms']} ms, "
          f"FPS = {results['yolov8n_detector_gpu']['mean_inference_fps']}")

    # 2. EasyOCR Plate OCR Benchmark: GPU vs CPU
    print("\n--- 2. EasyOCR Benchmark (GPU vs CPU) ---")
    test_plates = ["LEA-1234", "ICT-5678", "ABC-901", "KPK-2024"]
    plate_crops = []
    for text in test_plates:
        crop = np.ones((70, 240, 3), dtype=np.uint8) * 255
        cv2.putText(crop, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        plate_crops.append((text, crop))

    ocr_gpu = PlateOCR(gpu=True, n_reads=1)
    ocr_gpu.read(plate_crops[0][1])  # Warmup
    gpu_ocr_times = []
    for text, crop in plate_crops * 4:
        t0 = time.perf_counter()
        res = ocr_gpu.read(crop)
        gpu_ocr_times.append((time.perf_counter() - t0) * 1000)

    ocr_cpu = PlateOCR(gpu=False, n_reads=1)
    ocr_cpu.read(plate_crops[0][1])  # Warmup
    cpu_ocr_times = []
    for text, crop in plate_crops * 4:
        t0 = time.perf_counter()
        res_cpu = ocr_cpu.read(crop)
        cpu_ocr_times.append((time.perf_counter() - t0) * 1000)

    results["easyocr_plate_gpu"] = {
        "device": device_name,
        "median_ms": round(statistics.median(gpu_ocr_times), 2),
        "p95_ms": round(float(np.percentile(gpu_ocr_times, 95)), 2),
        "mean_fps": round(1000.0 / statistics.mean(gpu_ocr_times), 2),
    }
    results["easyocr_plate_cpu"] = {
        "device": "CPU (AMD Ryzen)",
        "median_ms": round(statistics.median(cpu_ocr_times), 2),
        "p95_ms": round(float(np.percentile(cpu_ocr_times, 95)), 2),
        "mean_fps": round(1000.0 / statistics.mean(cpu_ocr_times), 2),
    }
    ocr_speedup = round(results["easyocr_plate_cpu"]["median_ms"] / max(results["easyocr_plate_gpu"]["median_ms"], 0.1), 2)
    results["easyocr_speedup"] = f"{ocr_speedup}x"
    print(f"EasyOCR GPU: median = {results['easyocr_plate_gpu']['median_ms']} ms | CPU: median = {results['easyocr_plate_cpu']['median_ms']} ms | Speedup = {ocr_speedup}x")

    # 3. Face Detection & Feature Extraction (YuNet + SFace)
    print("\n--- 3. Face Pipeline (YuNet + SFace) ---")
    with tempfile.TemporaryDirectory() as td:
        store = EventStore(str(Path(td) / "bench.db"))
        face_engine = FaceEngine(store)
        lena = cv2.imread(str(ROOT / "logs/demo_data/opencv_lena.jpg"))
        if lena is None:
            lena = np.ones((360, 640, 3), dtype=np.uint8) * 128

        dets = face_engine.detect(lena)
        det_times, feat_times = [], []
        for _ in range(50):
            t0 = time.perf_counter()
            d = face_engine.detect(lena)
            det_times.append((time.perf_counter() - t0) * 1000)
            if d:
                t1 = time.perf_counter()
                f = face_engine.feature(lena, d[0])
                feat_times.append((time.perf_counter() - t1) * 1000)

        results["face_detection_yunet"] = {
            "median_ms": round(statistics.median(det_times), 2),
            "p95_ms": round(float(np.percentile(det_times, 95)), 2),
            "mean_fps": round(1000.0 / statistics.mean(det_times), 2),
        }
        if feat_times:
            results["face_recognition_sface"] = {
                "median_ms": round(statistics.median(feat_times), 2),
                "p95_ms": round(float(np.percentile(feat_times, 95)), 2),
                "mean_fps": round(1000.0 / statistics.mean(feat_times), 2),
            }
        print(f"YuNet Detection: median = {results['face_detection_yunet']['median_ms']} ms")
        if feat_times:
            print(f"SFace Feature: median = {results['face_recognition_sface']['median_ms']} ms")

    # 4. Sustained End-to-End Pipeline Latency Benchmark (100-frame warm-up + sustained measurement)
    print(f"\n--- 4. Sustained End-to-End Pipeline Latency Benchmark ({sustained_frames} frames) ---")
    with tempfile.TemporaryDirectory() as td:
        store = EventStore(str(Path(td) / "sustained.db"))
        runtime = SoftwareRuntime(store)
        pipeline = ReplayPipeline("scene", runtime, lambda evt: None, device=device)

        # 100-frame unrecorded warmup
        print("Executing 100-frame unrecorded warm-up...")
        for i in range(100):
            frame = frames[i % len(frames)]
            pipeline.process_frame(frame)

        # Sustained measurement run
        print(f"Executing {sustained_frames}-frame sustained measurement run...")
        e2e_latencies = []
        detector_latencies = []
        tracking_latencies = []

        for i in range(sustained_frames):
            frame = frames[i % len(frames)]
            t_ingress = time.perf_counter()
            res = pipeline.process_frame(frame)
            t_dispatch = time.perf_counter()

            total_ms = (t_dispatch - t_ingress) * 1000.0
            e2e_latencies.append(total_ms)
            stage_lat = pipeline.metrics.get("stage_latencies", {})
            if "detector_inference_ms" in stage_lat:
                detector_latencies.append(stage_lat["detector_inference_ms"])
            if "tracking_ms" in stage_lat:
                tracking_latencies.append(stage_lat["tracking_ms"])

        results["sustained_e2e_scene_pipeline"] = {
            "device": device_name,
            "warmup_frames": 100,
            "measured_frames": sustained_frames,
            "p50_ms": round(statistics.median(e2e_latencies), 2),
            "p90_ms": round(float(np.percentile(e2e_latencies, 90)), 2),
            "p95_ms": round(float(np.percentile(e2e_latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(e2e_latencies, 99)), 2),
            "mean_ms": round(statistics.mean(e2e_latencies), 2),
            "min_ms": round(min(e2e_latencies), 2),
            "max_ms": round(max(e2e_latencies), 2),
            "throughput_fps": round(1000.0 / statistics.mean(e2e_latencies), 2),
            "detector_inference_median_ms": round(statistics.median(detector_latencies), 2) if detector_latencies else None,
            "tracking_median_ms": round(statistics.median(tracking_latencies), 2) if tracking_latencies else None,
        }
        print(f"Sustained E2E Latency: p50 = {results['sustained_e2e_scene_pipeline']['p50_ms']} ms, "
              f"p95 = {results['sustained_e2e_scene_pipeline']['p95_ms']} ms, "
              f"p99 = {results['sustained_e2e_scene_pipeline']['p99_ms']} ms, "
              f"Throughput = {results['sustained_e2e_scene_pipeline']['throughput_fps']} FPS")

    # 5. Load CPU Baseline for Direct Head-to-Head Comparison
    cpu_baseline_file = ROOT / "logs/benchmarks/cpu_baseline.json"
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "gpu": device_name,
            "cpu": platform.processor(),
            "os": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": cuda_available,
        },
        "video": Path(video_path).name,
        "gpu_results": results,
    }

    if cpu_baseline_file.exists():
        cpu_data = json.loads(cpu_baseline_file.read_text(encoding="utf-8"))
        comparison["cpu_baseline"] = cpu_data.get("results", {})

        cpu_scene = cpu_data.get("results", {}).get("scene_detection_tracking", {})
        if cpu_scene and "median_ms" in cpu_scene:
            cpu_median = cpu_scene["median_ms"]
            gpu_e2e_median = results["sustained_e2e_scene_pipeline"]["p50_ms"]
            speedup = round(cpu_median / max(gpu_e2e_median, 0.01), 2)
            comparison["scene_pipeline_speedup"] = {
                "cpu_median_ms": cpu_median,
                "gpu_median_ms": gpu_e2e_median,
                "speedup_ratio": f"{speedup}x",
                "latency_reduction_percent": f"{round((1.0 - (gpu_e2e_median / cpu_median)) * 100, 1)}%",
            }
            print(f"\n=== OVERALL SCENE PIPELINE SPEEDUP ===")
            print(f"CPU Baseline Median: {cpu_median} ms \u2192 GPU Median: {gpu_e2e_median} ms")
            print(f"Speedup: {speedup}x ({comparison['scene_pipeline_speedup']['latency_reduction_percent']} latency reduction)")

    # Save output files
    benchmarks_dir = ROOT / "logs/benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    gpu_baseline_path = benchmarks_dir / "gpu_baseline.json"
    gpu_baseline_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved GPU baseline to: {gpu_baseline_path}")

    comparison_path = benchmarks_dir / "m02_gpu_vs_cpu_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved comparison to: {comparison_path}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(ROOT / "logs/demo_data/llvip_visible.mp4"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--sustained", type=int, default=300)
    args = parser.parse_args()

    run_gpu_benchmark(args.video, num_frames=args.frames, sustained_frames=args.sustained)
