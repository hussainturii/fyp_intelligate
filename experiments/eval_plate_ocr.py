"""Pakistani License Plate OCR Benchmark and Evaluation.
Evaluates EasyOCR performance, character error rates (CER), format validation,
and GPU vs CPU latency across curated Pakistani license plate patterns.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from module2_vehicle.plate_ocr import PlateOCR, _PLATE_PATTERNS, _OCR_FIXES


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def generate_synthetic_plate(text: str, province: str = "PUNJAB", bg_color=(240, 240, 240), noise: bool = False) -> np.ndarray:
    """Render a synthetic Pakistani license plate image with text, province header, and border."""
    width, height = 260, 80
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = bg_color

    # Draw border
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), (20, 20, 20), 2)

    # Draw province banner
    cv2.putText(img, province.upper(), (width // 2 - 35, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    # Draw main plate alphanumeric text
    # Clean text to uppercase
    cv2.putText(img, text.upper(), (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (10, 10, 10), 3, cv2.LINE_AA)

    if noise:
        # Add slight Gaussian noise to simulate camera grain
        gaussian = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + gaussian, 0, 255).astype(np.uint8)
        return noisy

    return img


def run_plate_evaluation():
    print("=== INTELLIGATE PAKISTANI LICENSE PLATE OCR EVALUATION ===")

    # Curate standard Pakistani test plates across provinces
    test_dataset = [
        # Islamabad Capital Territory (ICT)
        {"plate": "ICT-1234", "province": "ICT", "condition": "clean"},
        {"plate": "ISB-5678", "province": "ICT", "condition": "clean"},
        {"plate": "ICT-901",  "province": "ICT", "condition": "noisy"},
        {"plate": "ISB-2468", "province": "ICT", "condition": "clean"},
        {"plate": "ICT-3579", "province": "ICT", "condition": "noisy"},

        # Punjab
        {"plate": "LEA-1234", "province": "PUNJAB", "condition": "clean"},
        {"plate": "LXZ-5678", "province": "PUNJAB", "condition": "clean"},
        {"plate": "RIL-3456", "province": "PUNJAB", "condition": "noisy"},
        {"plate": "FD-7890",  "province": "PUNJAB", "condition": "clean"},
        {"plate": "MN-1122",  "province": "PUNJAB", "condition": "clean"},
        {"plate": "LEA-9876", "province": "PUNJAB", "condition": "noisy"},
        {"plate": "LXZ-1357", "province": "PUNJAB", "condition": "clean"},

        # Sindh
        {"plate": "ABC-123",  "province": "SINDH", "condition": "clean"},
        {"plate": "KHI-4567", "province": "SINDH", "condition": "clean"},
        {"plate": "AZ-890",   "province": "SINDH", "condition": "noisy"},
        {"plate": "KHI-8910", "province": "SINDH", "condition": "clean"},
        {"plate": "SND-1234", "province": "SINDH", "condition": "noisy"},

        # Khyber Pakhtunkhwa (KP)
        {"plate": "KPK-1234", "province": "KPK", "condition": "clean"},
        {"plate": "PES-5678", "province": "KPK", "condition": "clean"},
        {"plate": "KPK-9012", "province": "KPK", "condition": "noisy"},
    ]

    print(f"Total curated evaluation samples: {len(test_dataset)}")

    # Initialize PlateOCR on GPU and CPU
    ocr_gpu = PlateOCR(gpu=True, n_reads=1)
    ocr_cpu = PlateOCR(gpu=False, n_reads=1)

    eval_records = []
    exact_matches_gpu = 0
    total_cer_gpu = 0.0
    gpu_latencies = []
    cpu_latencies = []

    for item in test_dataset:
        ground_truth = item["plate"]
        province = item["province"]
        is_noisy = (item["condition"] == "noisy")

        plate_img = generate_synthetic_plate(ground_truth, province, noise=is_noisy)

        # GPU read
        t0 = time.perf_counter()
        res_gpu = ocr_gpu.read(plate_img)
        gpu_time = (time.perf_counter() - t0) * 1000
        gpu_latencies.append(gpu_time)

        # CPU read
        t0_cpu = time.perf_counter()
        res_cpu = ocr_cpu.read(plate_img)
        cpu_time = (time.perf_counter() - t0_cpu) * 1000
        cpu_latencies.append(cpu_time)

        # Normalization and CER calculation
        pred_clean = res_gpu.clean_text.replace(" ", "").replace("-", "")
        gt_clean = ground_truth.replace(" ", "").replace("-", "")

        dist = levenshtein_distance(pred_clean, gt_clean)
        cer = dist / max(len(gt_clean), 1)
        total_cer_gpu += cer

        is_exact = (pred_clean == gt_clean)
        if is_exact:
            exact_matches_gpu += 1

        eval_records.append({
            "ground_truth": ground_truth,
            "province": province,
            "condition": item["condition"],
            "prediction_gpu": res_gpu.clean_text,
            "confidence_gpu": round(res_gpu.confidence, 4),
            "is_valid_format": res_gpu.is_valid,
            "exact_match": is_exact,
            "edit_distance": dist,
            "cer": round(cer, 4),
            "latency_gpu_ms": round(gpu_time, 2),
            "latency_cpu_ms": round(cpu_time, 2),
        })

    num_samples = len(test_dataset)
    exact_match_rate = round((exact_matches_gpu / num_samples) * 100.0, 2)
    mean_cer = round((total_cer_gpu / num_samples) * 100.0, 2)
    character_accuracy = round(100.0 - mean_cer, 2)

    results = {
        "dataset_name": "Curated Pakistani License Plate Test Subset (ICT, Punjab, Sindh, KPK)",
        "sample_count": num_samples,
        "exact_match_accuracy_percent": exact_match_rate,
        "mean_character_error_rate_percent": mean_cer,
        "mean_character_accuracy_percent": character_accuracy,
        "gpu_latency": {
            "median_ms": round(statistics.median(gpu_latencies), 2),
            "p95_ms": round(float(np.percentile(gpu_latencies, 95)), 2),
            "mean_ms": round(statistics.mean(gpu_latencies), 2),
        },
        "cpu_latency": {
            "median_ms": round(statistics.median(cpu_latencies), 2),
            "p95_ms": round(float(np.percentile(cpu_latencies, 95)), 2),
            "mean_ms": round(statistics.mean(cpu_latencies), 2),
        },
        "speedup_ratio": round(statistics.median(cpu_latencies) / max(statistics.median(gpu_latencies), 0.1), 2),
        "eval_records": eval_records,
    }

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Exact Plate Match Accuracy: {exact_match_rate}%")
    print(f"Mean Character Error Rate (CER): {mean_cer}%")
    print(f"Character Accuracy: {character_accuracy}%")
    print(f"GPU Median Latency: {results['gpu_latency']['median_ms']} ms | CPU Median: {results['cpu_latency']['median_ms']} ms")
    print(f"Speedup: {results['speedup_ratio']}x")

    # Save results
    output_dir = ROOT / "logs/benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "plate_ocr_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to: {out_file}")

    return results


if __name__ == "__main__":
    run_plate_evaluation()
