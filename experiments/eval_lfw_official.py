"""Official LFW 6,000-Pair Verification Protocol for OpenCV YuNet + SFace.
Implements strict 10-fold cross-validation with out-of-fold threshold calibration
to prevent threshold leakage, explicit accounting of face detection errors (FDE),
and generates publication-quality distribution & ROC curve plots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.vision import FaceEngine


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return dot / max(norm_a * norm_b, 1e-12)


def parse_lfw_pairs(pairs_path: Path, lfw_dir: Path):
    """
    Parse official LFW pairs.txt format:
    Header: <num_folds> <pairs_per_fold>  (e.g., '10 300')
    Then for each fold:
      - pairs_per_fold lines of: <name> <img_idx1> <img_idx2> (genuine)
      - pairs_per_fold lines of: <name1> <img_idx1> <name2> <img_idx2> (impostor)
    """
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found at: {pairs_path}")
    if not lfw_dir.exists():
        raise FileNotFoundError(f"LFW image directory not found at: {lfw_dir}")

    lines = pairs_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError("Empty pairs.txt file")

    first_line = lines[0].strip().split()
    if len(first_line) == 2:
        num_folds = int(first_line[0])
        pairs_per_fold = int(first_line[1])
        data_lines = lines[1:]
    else:
        num_folds = 10
        pairs_per_fold = 300
        data_lines = lines

    folds = []
    current_line = 0
    total_lines = len(data_lines)

    actual_img_dir = (lfw_dir / "lfw") if (lfw_dir / "lfw").exists() and any((lfw_dir / "lfw").iterdir()) else lfw_dir

    def format_path(name: str, idx: int) -> Path:
        filename = f"{name}_{int(idx):04d}.jpg"
        return actual_img_dir / name / filename

    for fold_idx in range(num_folds):
        fold_pairs = []
        # Genuine pairs in this fold
        for _ in range(pairs_per_fold):
            if current_line >= total_lines:
                break
            parts = data_lines[current_line].strip().split()
            current_line += 1
            if len(parts) >= 3:
                name, i1, i2 = parts[0], int(parts[1]), int(parts[2])
                fold_pairs.append({
                    "is_same": True,
                    "img1_path": str(format_path(name, i1)),
                    "img2_path": str(format_path(name, i2)),
                    "id1": name,
                    "id2": name,
                })

        # Impostor pairs in this fold
        for _ in range(pairs_per_fold):
            if current_line >= total_lines:
                break
            parts = data_lines[current_line].strip().split()
            current_line += 1
            if len(parts) >= 4:
                n1, i1, n2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                fold_pairs.append({
                    "is_same": False,
                    "img1_path": str(format_path(n1, i1)),
                    "img2_path": str(format_path(n2, i2)),
                    "id1": n1,
                    "id2": n2,
                })

        folds.append(fold_pairs)

    return folds


def evaluate_lfw(lfw_dir: Path, pairs_path: Path, output_file: Path | None = None, plot_file: Path | None = None):
    print("=== INTELLIGATE OFFICIAL LFW FACE VERIFICATION PROTOCOL ===", flush=True)
    actual_img_dir = (lfw_dir / "lfw") if (lfw_dir / "lfw").exists() and any((lfw_dir / "lfw").iterdir()) else lfw_dir
    print(f"LFW Image Directory: {actual_img_dir}", flush=True)
    print(f"Protocol Pairs File: {pairs_path}", flush=True)

    store = EventStore(":memory:")
    face_engine = FaceEngine(store)

    folds = parse_lfw_pairs(pairs_path, lfw_dir)
    num_folds = len(folds)
    total_pairs = sum(len(f) for f in folds)
    print(f"Loaded {num_folds} folds containing {total_pairs} evaluation pairs.", flush=True)

    # Cache image embeddings to avoid re-extracting for repeated subjects
    embedding_cache = {}
    detection_failures = 0
    total_evaluated_pairs = 0

    def get_embedding(img_path_str: str):
        nonlocal detection_failures
        if img_path_str in embedding_cache:
            return embedding_cache[img_path_str]

        img = cv2.imread(img_path_str)
        if img is None:
            embedding_cache[img_path_str] = None
            return None

        dets = face_engine.detect(img)
        if not dets:
            detection_failures += 1
            embedding_cache[img_path_str] = None
            return None

        # Take largest detected face
        best_face = max(dets, key=lambda d: d[2] * d[3])
        emb = face_engine.feature(img, best_face)
        embedding_cache[img_path_str] = emb
        return emb

    fold_scores = []
    t_start = time.time()
    for f_idx, fold in enumerate(folds):
        t0 = time.time()
        scores = []
        labels = []
        fde_count = 0

        for pair in fold:
            total_evaluated_pairs += 1
            emb1 = get_embedding(pair["img1_path"])
            emb2 = get_embedding(pair["img2_path"])

            if emb1 is None or emb2 is None:
                # Face Detection Error (FDE): penalized as non-match (-1.0 similarity)
                fde_count += 1
                sim = -1.0
            else:
                sim = cosine_similarity(emb1, emb2)

            scores.append(sim)
            labels.append(1 if pair["is_same"] else 0)

        dt = time.time() - t0
        print(f"Fold {f_idx + 1:2d}/{num_folds} evaluated in {dt:5.2f}s | FDE in fold: {fde_count}", flush=True)

        fold_scores.append({
            "fold_idx": f_idx,
            "scores": np.array(scores, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32),
            "fde_count": fde_count,
        })

    total_eval_time = time.time() - t_start
    print(f"\nAll {total_pairs} pairs evaluated in {total_eval_time:.1f}s across {len(embedding_cache)} cached faces.", flush=True)

    # Strict 10-fold cross validation
    # Threshold is optimized strictly on the 9 training folds, tested on the remaining fold
    test_accuracies = []
    optimal_thresholds = []

    threshold_candidates = np.linspace(-0.2, 0.9, 1101)

    print("\n--- Out-of-Fold Threshold Calibration & Testing ---", flush=True)
    for test_idx in range(num_folds):
        # 1. Gather training scores from remaining 9 folds
        train_scores = np.concatenate([fold_scores[i]["scores"] for i in range(num_folds) if i != test_idx])
        train_labels = np.concatenate([fold_scores[i]["labels"] for i in range(num_folds) if i != test_idx])

        # 2. Find optimal threshold on training folds
        best_tau = 0.0
        best_train_acc = -1.0
        for tau in threshold_candidates:
            preds = (train_scores >= tau).astype(np.int32)
            acc = float(np.mean(preds == train_labels))
            if acc > best_train_acc:
                best_train_acc = acc
                best_tau = float(tau)

        # 3. Evaluate strictly on unseen held-out test fold
        test_s = fold_scores[test_idx]["scores"]
        test_l = fold_scores[test_idx]["labels"]
        test_preds = (test_s >= best_tau).astype(np.int32)
        test_acc = float(np.mean(test_preds == test_l))

        test_accuracies.append(test_acc)
        optimal_thresholds.append(best_tau)
        print(f"  Fold {test_idx + 1:2d}: Calibrated Tau = {best_tau:.4f} | Held-out Test Accuracy = {test_acc * 100:.2f}%", flush=True)

    mean_acc = float(np.mean(test_accuracies))
    std_acc = float(np.std(test_accuracies))
    sem_acc = std_acc / np.sqrt(num_folds)

    # Compute Global Pooled ROC Curve
    all_scores = np.concatenate([f["scores"] for f in fold_scores])
    all_labels = np.concatenate([f["labels"] for f in fold_scores])

    # Filter out FDE scores (-1.0) for pure embedding distribution metrics
    valid_mask = all_scores >= -0.5
    valid_scores = all_scores[valid_mask]
    valid_labels = all_labels[valid_mask]

    genuine_scores = valid_scores[valid_labels == 1]
    impostor_scores = valid_scores[valid_labels == 0]

    fpr_list, tpr_list = [], []
    eer = 1.0
    eer_tau = 0.5
    for tau in np.linspace(-0.2, 0.9, 1101):
        tpr = float(np.mean(genuine_scores >= tau)) if len(genuine_scores) > 0 else 0.0
        fpr = float(np.mean(impostor_scores >= tau)) if len(impostor_scores) > 0 else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        if abs(fpr - (1.0 - tpr)) < abs(eer - 0.5):
            eer = (fpr + (1.0 - tpr)) / 2.0
            eer_tau = float(tau)

    # Approximate AUC via trapezoidal integration
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    tpr_sorted = np.array(tpr_list)[sorted_indices]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    # Detailed Genuine and Impostor Distribution Summary
    dist_stats = {
        "genuine": {
            "count": int(len(genuine_scores)),
            "mean": round(float(np.mean(genuine_scores)), 4),
            "std": round(float(np.std(genuine_scores)), 4),
            "median": round(float(np.median(genuine_scores)), 4),
            "min": round(float(np.min(genuine_scores)), 4),
            "max": round(float(np.max(genuine_scores)), 4),
            "percentiles": {
                "p1": round(float(np.percentile(genuine_scores, 1)), 4),
                "p5": round(float(np.percentile(genuine_scores, 5)), 4),
                "p25": round(float(np.percentile(genuine_scores, 25)), 4),
                "p75": round(float(np.percentile(genuine_scores, 75)), 4),
                "p95": round(float(np.percentile(genuine_scores, 95)), 4),
            }
        },
        "impostor": {
            "count": int(len(impostor_scores)),
            "mean": round(float(np.mean(impostor_scores)), 4),
            "std": round(float(np.std(impostor_scores)), 4),
            "median": round(float(np.median(impostor_scores)), 4),
            "min": round(float(np.min(impostor_scores)), 4),
            "max": round(float(np.max(impostor_scores)), 4),
            "percentiles": {
                "p5": round(float(np.percentile(impostor_scores, 5)), 4),
                "p25": round(float(np.percentile(impostor_scores, 25)), 4),
                "p75": round(float(np.percentile(impostor_scores, 75)), 4),
                "p95": round(float(np.percentile(impostor_scores, 95)), 4),
                "p99": round(float(np.percentile(impostor_scores, 99)), 4),
            }
        }
    }

    results = {
        "benchmark": "Official LFW 6,000-Pair 10-Fold Verification Protocol",
        "dataset_source": "UMass Labeled Faces in the Wild (Official 6,000 pairs)",
        "detector": "OpenCV YuNet (face_detection_yunet_2023mar.onnx)",
        "recognizer": "OpenCV SFace (face_recognition_sface_2021dec.onnx)",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_pairs_evaluated": total_evaluated_pairs,
        "valid_pairs": int(np.sum(valid_mask)),
        "face_detection_errors_fde": detection_failures,
        "fde_rate_percent": round((detection_failures / (total_evaluated_pairs * 2)) * 100.0, 3),
        "ten_fold_results": {
            "mean_accuracy_percent": round(mean_acc * 100.0, 2),
            "std_dev_percent": round(std_acc * 100.0, 2),
            "std_error_mean_percent": round(sem_acc * 100.0, 2),
            "fold_accuracies_percent": [round(a * 100.0, 2) for a in test_accuracies],
            "calibrated_thresholds": [round(t, 4) for t in optimal_thresholds],
            "mean_calibrated_threshold": round(float(np.mean(optimal_thresholds)), 4),
        },
        "roc_metrics": {
            "auc": round(auc, 4),
            "eer_percent": round(eer * 100.0, 2),
            "eer_threshold": round(eer_tau, 4),
        },
        "score_distribution_statistics": dist_stats,
        "policy_invariant": "Advisory only (REVIEW_REQUIRED); autonomous gate actuation strictly prohibited",
    }

    print("\n=== LFW VERIFICATION FINAL RESULTS ===", flush=True)
    print(f"Mean 10-Fold Accuracy: {results['ten_fold_results']['mean_accuracy_percent']}% \u00b1 {results['ten_fold_results']['std_error_mean_percent']}%", flush=True)
    print(f"Mean Calibrated Threshold: {results['ten_fold_results']['mean_calibrated_threshold']}", flush=True)
    print(f"AUC: {results['roc_metrics']['auc']} | EER: {results['roc_metrics']['eer_percent']}% (at tau={results['roc_metrics']['eer_threshold']})", flush=True)
    print(f"Genuine Scores: mean={dist_stats['genuine']['mean']}, std={dist_stats['genuine']['std']}", flush=True)
    print(f"Impostor Scores: mean={dist_stats['impostor']['mean']}, std={dist_stats['impostor']['std']}", flush=True)
    print(f"Face Detection Failures (FDE): {detection_failures} ({results['fde_rate_percent']}%)", flush=True)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved machine-readable results to: {output_file}", flush=True)

    # Generate Publication Plot
    if plot_file is None and output_file:
        plot_file = output_file.with_suffix(".png")

    if plot_file:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Score distributions
        ax1.hist(genuine_scores, bins=60, density=True, alpha=0.65, color="seagreen", label="Genuine Pairs (Same Identity)")
        ax1.hist(impostor_scores, bins=60, density=True, alpha=0.65, color="crimson", label="Impostor Pairs (Different Identity)")
        mean_tau = results["ten_fold_results"]["mean_calibrated_threshold"]
        ax1.axvline(mean_tau, color="darkblue", linestyle="--", linewidth=2, label=f"Calibrated Tau ({mean_tau})")
        ax1.axvline(eer_tau, color="purple", linestyle=":", linewidth=2, label=f"EER Tau ({eer_tau:.3f})")
        ax1.set_title("LFW Genuine vs Impostor Score Distributions (YuNet + SFace)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Cosine Similarity", fontsize=11)
        ax1.set_ylabel("Probability Density", fontsize=11)
        ax1.legend(loc="upper left", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. ROC Curve
        ax2.plot(fpr_sorted, tpr_sorted, color="royalblue", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
        ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.50)")
        ax2.scatter([eer], [1.0 - eer], color="darkred", s=70, zorder=5, label=f"EER = {eer * 100:.2f}%")
        ax2.set_title("LFW Official Verification ROC Curve", fontsize=12, fontweight="bold")
        ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax2.set_ylabel("True Positive Rate (TPR)", fontsize=11)
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file, dpi=200)
        plt.close(fig)
        print(f"Saved publication chart to: {plot_file}", flush=True)

    return results


def check_dataset_status(lfw_dir: Path, pairs_path: Path):
    """Inspect and report whether the official LFW dataset is present locally."""
    actual_img_dir = (lfw_dir / "lfw") if (lfw_dir / "lfw").exists() and any((lfw_dir / "lfw").iterdir()) else lfw_dir
    lfw_ok = actual_img_dir.exists() and len(list(actual_img_dir.glob("*/*.jpg"))) > 1000
    pairs_ok = pairs_path.exists() and pairs_path.stat().st_size > 1000

    return {
        "lfw_directory_exists": lfw_ok,
        "pairs_file_exists": pairs_ok,
        "status": "READY" if (lfw_ok and pairs_ok) else "BLOCKED",
        "lfw_path": str(actual_img_dir),
        "pairs_path": str(pairs_path),
        "image_count": len(list(actual_img_dir.glob("*/*.jpg"))) if lfw_ok else 0,
        "download_instructions": {
            "lfw_archive_url": "https://ndownloader.figshare.com/files/5976018",
            "pairs_url": "https://ndownloader.figshare.com/files/5976006",
            "approximate_size": "~173 MB (lfw.tgz), ~150 KB (pairs.txt)",
            "license": "Non-commercial, academic research use (UMass Amherst)",
            "required_target_dir": str(lfw_dir.resolve()),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official LFW Verification Protocol for YuNet + SFace")
    parser.add_argument("--lfw-dir", default=str(ROOT / "data/lfw"), help="Path to unpacked LFW images")
    parser.add_argument("--pairs-file", default=str(ROOT / "data/lfw/pairs.txt"), help="Path to official pairs.txt")
    parser.add_argument("--output", default=str(ROOT / "logs/benchmarks/m03a_lfw_official_results.json"))
    parser.add_argument("--plot", default=str(ROOT / "logs/benchmarks/m03a_lfw_roc_distribution.png"))
    parser.add_argument("--check-only", action="store_true", help="Only verify dataset availability")
    args = parser.parse_args()

    lfw_p = Path(args.lfw_dir)
    pairs_p = Path(args.pairs_file)

    status = check_dataset_status(lfw_p, pairs_p)
    if args.check_only or status["status"] == "BLOCKED":
        print("=== LFW DATASET READINESS AUDIT ===", flush=True)
        print(f"Status: {status['status']}", flush=True)
        print(f"LFW Image Directory Present: {status['lfw_directory_exists']} ({status['image_count']} images at {status['lfw_path']})", flush=True)
        print(f"Pairs Protocol File Present: {status['pairs_file_exists']} ({status['pairs_path']})", flush=True)
        if status["status"] == "BLOCKED":
            print("\nDATASET ACQUISITION BLOCKER:", flush=True)
            print("The official LFW verification dataset is not currently in the workspace.", flush=True)
            sys.exit(2)
        if args.check_only:
            sys.exit(0)

    evaluate_lfw(lfw_p, pairs_p, Path(args.output), Path(args.plot))
