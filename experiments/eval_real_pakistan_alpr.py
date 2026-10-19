"""Real Pakistani License Plate Localization, OCR and Active Pipeline Benchmark.
Evaluates:
1. Plate Localization Precision, Recall, and mIoU on real Pakistani vehicle photos.
2. EasyOCR transcription performance on genuine vehicle crops.
3. Fair CPU vs GPU active ALPR workload benchmark on real road vehicle frames.
4. Generates an auditable human-in-the-loop OCR verification manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from module2_vehicle.plate_detector import PlateDetector
from module2_vehicle.plate_ocr import PlateOCR
from module2_vehicle.database.vehicle_db import VehicleDatabase
from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime


def compute_iou(boxA: list[int], boxB: list[int]) -> float:
    """Compute IoU between [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def yolo_to_xyxy(yolo_box: list[float], img_w: int, img_h: int) -> list[int]:
    """Convert YOLO format [x_c, y_c, w, h] normalized to [x1, y1, x2, y2] pixels."""
    xc, yc, w, h = yolo_box
    x1 = int((xc - w / 2.0) * img_w)
    y1 = int((yc - h / 2.0) * img_h)
    x2 = int((xc + w / 2.0) * img_w)
    y2 = int((yc + h / 2.0) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def load_partition(dataset_dir: Path, partition: str = "test"):
    """Load images and matching ground-truth bounding boxes."""
    imgs_dir = dataset_dir / partition / "images"
    lbls_dir = dataset_dir / partition / "labels"

    if not imgs_dir.exists():
        return []

    samples = []
    img_files = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpeg")))

    for img_p in img_files:
        lbl_p = lbls_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        gt_boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        pass
        samples.append({
            "image_path": img_p,
            "label_path": lbl_p,
            "yolo_boxes": gt_boxes,
        })
    return samples


def evaluate_localization(samples: list[dict], detector: PlateDetector):
    """Evaluate plate localization IoU, Precision@50, Recall@50."""
    total_gt = 0
    total_det = 0
    true_positives_50 = 0
    iou_scores = []
    latencies = []

    for s in samples:
        img = cv2.imread(str(s["image_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_xyxy = [yolo_to_xyxy(b, w, h) for b in s["yolo_boxes"]]
        total_gt += len(gt_xyxy)

        t0 = time.perf_counter()
        preds = detector.detect(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        total_det += len(preds)

        # Match detections to ground truth
        matched_gt = set()
        for p in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_xyxy):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(list(p.bbox), g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.50 and best_gt_idx >= 0:
                true_positives_50 += 1
                matched_gt.add(best_gt_idx)
                iou_scores.append(best_iou)
            elif best_iou > 0:
                iou_scores.append(best_iou)

    precision = true_positives_50 / total_det if total_det > 0 else 0.0
    recall = true_positives_50 / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    return {
        "total_images_evaluated": len(samples),
        "total_ground_truth_plates": total_gt,
        "total_detected_plates": total_det,
        "true_positives_iou50": true_positives_50,
        "precision_iou50": round(precision * 100.0, 2),
        "recall_iou50": round(recall * 100.0, 2),
        "f1_score_iou50": round(f1 * 100.0, 2),
        "mean_iou_on_hits": round(mean_iou, 4),
        "avg_detector_latency_ms": round(avg_lat, 2),
    }


def evaluate_real_alpr():
    print("=== INTELLIGATE REAL PAKISTANI LICENSE PLATE EVALUATION ===", flush=True)

    dataset_a_dir = ROOT / "data/pakistan_plates/dataset_a"
    dataset_b_dir = ROOT / "data/pakistan_plates/dataset_b"

    samples_a_test = load_partition(dataset_a_dir, "test")
    samples_a_valid = load_partition(dataset_a_dir, "valid")
    samples_b_test = load_partition(dataset_b_dir, "test")

    print(f"Loaded Dataset A: {len(samples_a_test)} test, {len(samples_a_valid)} valid samples (Outdoor Vehicle Scenes)", flush=True)
    print(f"Loaded Dataset B: {len(samples_b_test)} test samples (Close-Cropped Plates)", flush=True)

    # 1. Initialize models
    print("\n[INIT] Initializing PlateDetector and EasyOCR (GPU)...", flush=True)
    detector_gpu = PlateDetector(device="cuda:0", allow_download=False)
    ocr_gpu = PlateOCR(gpu=True, n_reads=1)

    detector_cpu = PlateDetector(device="cpu", allow_download=False)
    ocr_cpu = PlateOCR(gpu=False, n_reads=1)

    # 2. Evaluate Plate Localization on Dataset A (Real Vehicle Scenes)
    print("\n--- Evaluating Plate Localization on Dataset A (Outdoor Vehicle Scenes) ---", flush=True)
    all_a_eval_samples = samples_a_test + samples_a_valid
    loc_results_a = evaluate_localization(all_a_eval_samples, detector_gpu)
    print(f"Dataset A Localization (P@50): {loc_results_a['precision_iou50']}% | Recall@50: {loc_results_a['recall_iou50']}% | F1: {loc_results_a['f1_score_iou50']}% | mIoU: {loc_results_a['mean_iou_on_hits']}", flush=True)

    # 3. Evaluate Plate Localization on Dataset B
    print("\n--- Evaluating Plate Localization on Dataset B (Secondary Benchmark) ---", flush=True)
    loc_results_b = evaluate_localization(samples_b_test, detector_gpu)
    print(f"Dataset B Localization (P@50): {loc_results_b['precision_iou50']}% | Recall@50: {loc_results_b['recall_iou50']}% | F1: {loc_results_b['f1_score_iou50']}% | mIoU: {loc_results_b['mean_iou_on_hits']}", flush=True)

    # 4. OCR Evaluation & Annotation Manifest Creation
    print("\n--- Processing OCR on Real Vehicle Plate Crops & Generating Human Annotation Manifest ---", flush=True)
    crop_dir = ROOT / "data/pakistan_plates/verified_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = ROOT / "data/pakistan_plates/ocr_annotation_manifest.csv"
    manifest_rows = []

    ocr_predictions = []
    ocr_latencies_gpu = []
    ocr_latencies_cpu = []

    # Process up to 50 representative real Pakistani vehicle scenes for detailed OCR audit
    eval_subset = all_a_eval_samples[:50]
    for idx, s in enumerate(eval_subset):
        img = cv2.imread(str(s["image_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        if not s["yolo_boxes"]:
            continue

        box = yolo_to_xyxy(s["yolo_boxes"][0], w, h)
        crop = img[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        crop_fname = f"crop_sample_{idx:03d}_{s['image_path'].stem}.jpg"
        crop_path = crop_dir / crop_fname
        cv2.imwrite(str(crop_path), crop)

        # Run OCR GPU
        t0 = time.perf_counter()
        res_gpu = ocr_gpu.read(crop)
        ocr_latencies_gpu.append((time.perf_counter() - t0) * 1000.0)

        # Run OCR CPU for latency comparison
        t0 = time.perf_counter()
        res_cpu = ocr_cpu.read(crop)
        ocr_latencies_cpu.append((time.perf_counter() - t0) * 1000.0)

        text_gpu = res_gpu.clean_text or res_gpu.raw_text
        conf_gpu = res_gpu.confidence

        ocr_predictions.append({
            "image": s["image_path"].name,
            "crop_file": crop_fname,
            "raw_text": res_gpu.raw_text,
            "clean_text": res_gpu.clean_text,
            "confidence": round(float(conf_gpu), 3),
        })

        manifest_rows.append({
            "sample_id": idx + 1,
            "source_image": s["image_path"].name,
            "crop_relative_path": f"data/pakistan_plates/verified_crops/{crop_fname}",
            "plate_bbox_xyxy": f"[{box[0]},{box[1]},{box[2]},{box[3]}]",
            "easyocr_predicted_text": text_gpu,
            "easyocr_confidence": round(float(conf_gpu), 3),
            "human_verified_gt_text": "",  # To be verified by human reviewer
            "jurisdiction_province": "",   # ICT / PUNJAB / SINDH / KP
            "verified_status": "PENDING_HUMAN_REVIEW",
        })

    # Save CSV manifest
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "source_image", "crop_relative_path", "plate_bbox_xyxy",
            "easyocr_predicted_text", "easyocr_confidence", "human_verified_gt_text",
            "jurisdiction_province", "verified_status"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Generated OCR Annotation Manifest: {manifest_csv} ({len(manifest_rows)} crops logged)", flush=True)

    # 5. Active Pipeline Workload Benchmark on Genuine Road Vehicle Frames
    print("\n--- Benchmarking Active ALPR Workload on Real Pakistani Vehicle Frames (CPU vs GPU) ---", flush=True)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = EventStore(str(Path(td) / "eval.db"))
        runtime = SoftwareRuntime(store)
        vdb = VehicleDatabase(str(Path(td) / "vdb.h5"), str(Path(td) / "unknown.h5"))

        # Whitelist two representative recognized plates to test gate policy
        whitelisted = ["LEA1234", "ICT5678", "LED5678"]
        for p in whitelisted:
            vdb.register(p, f"owner_{p}", "faculty")

        # Benchmark 40 cycles on real vehicle frames
        benchmark_images = [cv2.imread(str(s["image_path"])) for s in eval_subset[:40]]
        benchmark_images = [img for img in benchmark_images if img is not None]

        # GPU Run
        gpu_cycle_latencies = []
        gpu_ocr_invocations = 0
        gpu_gate_allows = 0
        gpu_gate_denies = 0

        for frame in benchmark_images:
            t0 = time.perf_counter()
            plates = detector_gpu.detect(frame)
            if plates:
                gpu_ocr_invocations += 1
                ocr_res = ocr_gpu.read(plates[0].crop)
                plate_txt = ocr_res.clean_text or ocr_res.raw_text
                match = vdb.lookup_by_plate(plate_txt) if plate_txt else None
                # Enforce gate invariant: exact match required
                if match and match.get("method") == "exact_plate":
                    obs = runtime.observe(frame, [match["plate"]], source_verified=True, trusted=True)
                    if obs["decision"] == "ALLOW":
                        gpu_gate_allows += 1
                    else:
                        gpu_gate_denies += 1
                else:
                    gpu_gate_denies += 1
            else:
                gpu_gate_denies += 1
            gpu_cycle_latencies.append((time.perf_counter() - t0) * 1000.0)

        # CPU Run
        cpu_cycle_latencies = []
        cpu_ocr_invocations = 0

        for frame in benchmark_images:
            t0 = time.perf_counter()
            plates = detector_cpu.detect(frame)
            if plates:
                cpu_ocr_invocations += 1
                ocr_res = ocr_cpu.read(plates[0].crop)
                plate_txt = ocr_res.clean_text or ocr_res.raw_text
                match = vdb.lookup_by_plate(plate_txt) if plate_txt else None
                if match and match.get("method") == "exact_plate":
                    obs = runtime.observe(frame, [match["plate"]], source_verified=True, trusted=True)
            cpu_cycle_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_gpu_lat = float(np.mean(gpu_cycle_latencies))
    avg_cpu_lat = float(np.mean(cpu_cycle_latencies))
    fps_gpu = 1000.0 / avg_gpu_lat if avg_gpu_lat > 0 else 0.0
    fps_cpu = 1000.0 / avg_cpu_lat if avg_cpu_lat > 0 else 0.0
    speedup = avg_cpu_lat / avg_gpu_lat if avg_gpu_lat > 0 else 1.0

    print(f"\nReal Vehicle Traffic Active Workload:")
    print(f"  GPU Pipeline Latency: {avg_gpu_lat:.2f} ms ({fps_gpu:.2f} FPS) | OCR Calls: {gpu_ocr_invocations}")
    print(f"  CPU Pipeline Latency: {avg_cpu_lat:.2f} ms ({fps_cpu:.2f} FPS) | OCR Calls: {cpu_ocr_invocations}")
    print(f"  Actual Hardware Speedup on Real Road Traffic: {speedup:.2f}x")
    print(f"  Gate Decisions: {gpu_gate_allows} ALLOW (Whitelisted Exact Match), {gpu_gate_denies} DENY/HOLD")

    final_results = {
        "benchmark": "Real Pakistani Vehicle License Plate Localization & Active ALPR Audit",
        "dataset_a_source": "Roboflow Universe pakistani-number-plates-4dyfe/6 (CC BY 4.0)",
        "dataset_b_source": "Roboflow Universe pakistani_license_number_plates/1 (CC BY 4.0)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "localization_dataset_a": loc_results_a,
        "localization_dataset_b": loc_results_b,
        "ocr_performance": {
            "evaluated_real_crops": len(eval_subset),
            "avg_easyocr_gpu_latency_ms": round(float(np.mean(ocr_latencies_gpu)), 2),
            "avg_easyocr_cpu_latency_ms": round(float(np.mean(ocr_latencies_cpu)), 2),
            "sample_transcriptions": ocr_predictions[:10],
            "text_ground_truth_status": "MISSING_IN_SOURCE_DATASET",
            "safeguard_policy": "NO_FABRICATION: Plate localization evaluated mathematically; OCR CER requires human verification of generated manifest",
            "manifest_file": str(manifest_csv),
        },
        "active_workload_real_traffic": {
            "evaluated_cycles": len(benchmark_images),
            "gpu_mean_latency_ms": round(avg_gpu_lat, 2),
            "gpu_sustained_fps": round(fps_gpu, 2),
            "cpu_mean_latency_ms": round(avg_cpu_lat, 2),
            "cpu_sustained_fps": round(fps_cpu, 2),
            "speedup_factor": round(speedup, 2),
            "ocr_invocation_rate_percent": round((gpu_ocr_invocations / len(benchmark_images)) * 100.0, 1),
            "gate_allow_count": gpu_gate_allows,
            "gate_deny_count": gpu_gate_denies,
            "exact_match_security_invariant_enforced": True,
        }
    }

    out_json = ROOT / "logs/benchmarks/m03a_real_pakistan_alpr_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable ALPR audit results to: {out_json}", flush=True)

    return final_results


if __name__ == "__main__":
    evaluate_real_alpr()
