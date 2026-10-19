"""
experiments/eval_m03d_vehicle_reid.py
======================================
Scientific Evaluation & Benchmarking for Milestone 3D:
Deep Metric Vehicle Re-Identification (OSNet-AIN) vs. Color Histogram Baseline.

Produces:
1. Extraction latency statistics (warmup, 100 iterations, mean, median, std, 95th pct, FPS).
2. FAISS gallery search latency across varying gallery sizes.
3. Pairwise cosine similarity distributions on real local vehicle crops (same-vehicle vs. different-vehicle).
4. Margin of separation and discrimination metrics comparing OSNet-AIN against the 768-D color histogram.
5. Strict attribution of published VeRi-776 benchmarks and honest notation of local multi-camera metrics.

Saves raw metrics to: logs/benchmarks/m03d_vehicle_reid_results.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / "logs" / "benchmarks"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = LOG_DIR / "m03d_vehicle_reid_results.json"

from module2_vehicle.database.vehicle_db import VehicleDatabase
from module2_vehicle.vehicle_reid import (
    VehicleReID,
    color_histogram_baseline,
)


def benchmark_extraction_latency(reid: VehicleReID, crop: np.ndarray, n_runs: int = 100) -> Dict:
    """Benchmark extraction latency with warmup."""
    # 10 warmup runs
    for _ in range(10):
        _ = reid.extract_embedding(crop)

    times_deep = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = reid.extract_embedding(crop)
        t1 = time.perf_counter()
        times_deep.append((t1 - t0) * 1000)

    # Benchmark histogram baseline
    for _ in range(10):
        _ = color_histogram_baseline(crop)

    times_hist = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = color_histogram_baseline(crop)
        t1 = time.perf_counter()
        times_hist.append((t1 - t0) * 1000)

    return {
        "deep_reid_osnet": {
            "n_runs": n_runs,
            "mean_ms": round(statistics.mean(times_deep), 2),
            "median_ms": round(statistics.median(times_deep), 2),
            "std_ms": round(statistics.stdev(times_deep), 2),
            "p95_ms": round(float(np.percentile(times_deep, 95)), 2),
            "min_ms": round(min(times_deep), 2),
            "max_ms": round(max(times_deep), 2),
            "throughput_fps": round(1000.0 / statistics.mean(times_deep), 1),
            "device": reid.active_target,
            "dimensions": reid.embedding_dim,
        },
        "color_histogram_baseline": {
            "n_runs": n_runs,
            "mean_ms": round(statistics.mean(times_hist), 2),
            "median_ms": round(statistics.median(times_hist), 2),
            "std_ms": round(statistics.stdev(times_hist), 2),
            "p95_ms": round(float(np.percentile(times_hist, 95)), 2),
            "min_ms": round(min(times_hist), 2),
            "max_ms": round(max(times_hist), 2),
            "throughput_fps": round(1000.0 / statistics.mean(times_hist), 1),
            "device": "cpu",
            "dimensions": 768,
        },
    }


def benchmark_faiss_search(reid: VehicleReID, gallery_sizes: List[int] = [1, 10, 50, 100, 500]) -> Dict:
    """Benchmark FAISS search latency across different gallery sizes."""
    import tempfile
    results = {}
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    for n in gallery_sizes:
        with tempfile.TemporaryDirectory() as td:
            db = VehicleDatabase(
                db_path=str(Path(td) / "v.h5"),
                unknown_db_path=str(Path(td) / "u.h5"),
                embedding_dim=512,
            )
            # Register n vehicles
            for i in range(n):
                emb = np.random.randn(512).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                db.register(f"TEST{i:04d}", f"Owner_{i}", embedding=emb)

            # Warmup
            for _ in range(5):
                _ = db.search_by_embedding(query_emb, k=5)

            # Measure 100 searches
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = db.search_by_embedding(query_emb, k=5)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            results[f"gallery_{n}"] = {
                "mean_ms": round(statistics.mean(times), 4),
                "median_ms": round(statistics.median(times), 4),
                "throughput_qps": round(1000.0 / statistics.mean(times), 1),
            }

    return results


def evaluate_pairwise_distributions(reid: VehicleReID, sample_crops: List[Path]) -> Dict:
    """
    Compare pairwise cosine similarity for same-vehicle and different-vehicle pairs
    between OSNet-AIN 512-D and Color Histogram 768-D.
    """
    same_deep, diff_deep = [], []
    same_hist, diff_hist = [], []

    loaded_crops = []
    for p in sample_crops[:30]:  # Up to 30 crops
        img = cv2.imread(str(p))
        if img is not None and img.shape[0] >= 40 and img.shape[1] >= 40:
            loaded_crops.append(img)

    if len(loaded_crops) < 2:
        return {"status": "insufficient_crops"}

    # Extract all embeddings
    deep_embs = [reid.extract_embedding(c) for c in loaded_crops]
    hist_embs = [color_histogram_baseline(c) for c in loaded_crops]

    # 1. Same-vehicle pairs (simulated by multi-scale sub-crops / perturbations of same vehicle)
    for i, img in enumerate(loaded_crops):
        h, w = img.shape[:2]
        # Slight crop shifts
        crop_a = img[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
        crop_b = cv2.convertScaleAbs(crop_a, alpha=0.9, beta=5)  # slight illumination shift

        emb_orig = deep_embs[i]
        emb_sub = reid.extract_embedding(crop_b)
        same_deep.append(float(np.dot(emb_orig, emb_sub)))

        hist_orig = hist_embs[i]
        hist_sub = color_histogram_baseline(crop_b)
        same_hist.append(float(np.dot(hist_orig, hist_sub)))

    # 2. Different-vehicle pairs (cross pairs between distinct vehicle crops)
    for i in range(len(loaded_crops)):
        for j in range(i + 1, min(i + 6, len(loaded_crops))):
            diff_deep.append(float(np.dot(deep_embs[i], deep_embs[j])))
            diff_hist.append(float(np.dot(hist_embs[i], hist_embs[j])))

    mu_same_deep = statistics.mean(same_deep)
    mu_diff_deep = statistics.mean(diff_deep)
    margin_deep = mu_same_deep - mu_diff_deep

    mu_same_hist = statistics.mean(same_hist)
    mu_diff_hist = statistics.mean(diff_hist)
    margin_hist = mu_same_hist - mu_diff_hist

    return {
        "evaluation_protocol": "Synthetic Intra-Image Augmentation Consistency & Cross-Vehicle Pairing (Real Vehicle-Body Crops)",
        "scientific_clarification": (
            "Per Milestone 3D.1 review, intra-image perturbation tests evaluate invariance to geometric scale and photometric shift, "
            "not genuine cross-camera multi-view identity Re-ID. Prior evaluation using license plate crops is formally invalidated. "
            "Multi-camera vehicle identity metrics (Rank-1 / mAP) remain NOT EVALUATED."
        ),
        "deep_reid_osnet_512d": {
            "n_same_pairs": len(same_deep),
            "n_diff_pairs": len(diff_deep),
            "test_type": "Synthetic Intra-Image Augmentation Consistency (Intra-vehicle Perturbation)",
            "same_vehicle_similarity": {
                "mean": round(mu_same_deep, 4),
                "median": round(statistics.median(same_deep), 4),
                "std": round(statistics.stdev(same_deep), 4),
                "min": round(min(same_deep), 4),
                "max": round(max(same_deep), 4),
            },
            "different_vehicle_similarity": {
                "mean": round(mu_diff_deep, 4),
                "median": round(statistics.median(diff_deep), 4),
                "std": round(statistics.stdev(diff_deep), 4),
                "min": round(min(diff_deep), 4),
                "max": round(max(diff_deep), 4),
            },
            "separation_margin": round(margin_deep, 4),
        },
        "color_histogram_768d": {
            "n_same_pairs": len(same_hist),
            "n_diff_pairs": len(diff_hist),
            "test_type": "Synthetic Intra-Image Augmentation Consistency (Intra-vehicle Perturbation)",
            "same_vehicle_similarity": {
                "mean": round(mu_same_hist, 4),
                "median": round(statistics.median(same_hist), 4),
                "std": round(statistics.stdev(same_hist), 4),
                "min": round(min(same_hist), 4),
                "max": round(max(same_hist), 4),
            },
            "different_vehicle_similarity": {
                "mean": round(mu_diff_hist, 4),
                "median": round(statistics.median(diff_hist), 4),
                "std": round(statistics.stdev(diff_hist), 4),
                "min": round(min(diff_hist), 4),
                "max": round(max(diff_hist), 4),
            },
            "separation_margin": round(margin_hist, 4),
        },
        "margin_improvement_ratio": round(margin_deep / max(margin_hist, 1e-6), 2),
    }


def main():
    print("=== INTELLIGATE MILESTONE 3D: VEHICLE RE-ID EVALUATION ===")
    reid = VehicleReID(device="cpu")
    print(f"Loaded: {reid.model_id} | Dim: {reid.embedding_dim}")

    # Locate test crops: prioritize verified vehicle body crops extracted from road scenes
    vehicle_body_dir = ROOT / "data" / "pakistan_plates" / "vehicle_body_crops"
    if vehicle_body_dir.exists() and len(list(vehicle_body_dir.glob("*.jpg"))) >= 10:
        sample_crops = sorted(list(vehicle_body_dir.glob("*.jpg")))
        crop_type = "Verified Vehicle-Body Crops (YOLOv8 detected from Pakistani road scenes)"
    else:
        sample_crops = sorted(list(ROOT.glob("data/pakistan_plates/vehicle_body_crops/*.jpg")))
        crop_type = "Vehicle-Body Crops"
    print(f"Found {len(sample_crops)} {crop_type}.")

    test_img = cv2.imread(str(sample_crops[0])) if sample_crops else np.zeros((208, 208, 3), dtype=np.uint8)

    print("\n[1/3] Benchmarking Extraction Latency (100 iterations)...")
    latency_results = benchmark_extraction_latency(reid, test_img, n_runs=100)
    print(f"Deep Re-ID (CPU): {latency_results['deep_reid_osnet']['median_ms']} ms "
          f"({latency_results['deep_reid_osnet']['throughput_fps']} FPS)")
    print(f"Color Histogram (CPU): {latency_results['color_histogram_baseline']['median_ms']} ms "
          f"({latency_results['color_histogram_baseline']['throughput_fps']} FPS)")

    print("\n[2/3] Benchmarking FAISS Gallery Search Latency...")
    faiss_results = benchmark_faiss_search(reid)
    for g, res in faiss_results.items():
        print(f"  {g}: {res['median_ms']} ms ({res['throughput_qps']} QPS)")

    print("\n[3/3] Evaluating Pairwise Cosine Similarity Distributions...")
    pairwise_results = evaluate_pairwise_distributions(reid, sample_crops)
    if "deep_reid_osnet_512d" in pairwise_results:
        d_same = pairwise_results["deep_reid_osnet_512d"]["same_vehicle_similarity"]["mean"]
        d_diff = pairwise_results["deep_reid_osnet_512d"]["different_vehicle_similarity"]["mean"]
        d_margin = pairwise_results["deep_reid_osnet_512d"]["separation_margin"]

        h_same = pairwise_results["color_histogram_768d"]["same_vehicle_similarity"]["mean"]
        h_diff = pairwise_results["color_histogram_768d"]["different_vehicle_similarity"]["mean"]
        h_margin = pairwise_results["color_histogram_768d"]["separation_margin"]

        print(f"OSNet-AIN: Same={d_same:.4f}, Diff={d_diff:.4f}, Separation Margin={d_margin:.4f}")
        print(f"Histogram: Same={h_same:.4f}, Diff={h_diff:.4f}, Separation Margin={h_margin:.4f}")
        print(f"Discriminative Margin Advantage: {pairwise_results['margin_improvement_ratio']}x")

    full_report = {
        "milestone": "3D",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "name": "osnet_ain_x1_0_vehicle_reid.onnx",
            "model_id": reid.model_id,
            "architecture": "OSNet-AIN-x1.0",
            "parameter_count": "2.183M",
            "gflops": 2.643,
            "embedding_dimension": 512,
            "input_resolution": [208, 208],
            "input_channels": "RGB",
            "license": "MIT",
            "provenance": "Intel Open Model Zoo (vehicle-reid-0001)",
            "published_veri776_benchmarks": {
                "rank_1": 0.9631,
                "mAP": 0.8515,
                "dataset": "VeRi-776 (576 train IDs, 200 test IDs, 20 cameras)",
                "source": "Intel Open Model Zoo documentation (vehicle-reid-0001/README.md)"
            },
            "local_multicamera_traffic_benchmark": {
                "rank_1": "NOT EVALUATED (Requires multi-camera benchmark dataset)",
                "mAP": "NOT EVALUATED (Requires multi-camera benchmark dataset)",
                "reason": "Local dataset contains single-camera snapshots without verified cross-camera identity tracking; no synthetic identities fabricated."
            }
        },
        "latency_benchmark": latency_results,
        "gallery_search_benchmark": faiss_results,
        "pairwise_similarity_evaluation": pairwise_results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[DONE] Saved evaluation report to: {OUT_JSON}")


if __name__ == "__main__":
    main()
