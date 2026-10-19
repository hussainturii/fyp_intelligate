"""
Train YOLOv8n License Plate Detector on NVIDIA RTX 4060 Ti GPU.
IntelliGate Milestone 3B.
- Pretrained YOLOv8n initialization
- Clean combined Pakistani license plate dataset
- Early stopping with patience=12
- Full checkpointing (best.pt and last.pt)
- VRAM and latency telemetry logging
- Reproducible seed=42
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    start_time = time.time()
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    data_yaml = root / "data" / "pakistan_plates" / "combined_clean" / "dataset.yaml"
    init_weights = root / "yolov8n.pt"

    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found at {data_yaml}")
        sys.exit(1)

    if not init_weights.exists():
        print(f"ERROR: Base weights not found at {init_weights}")
        sys.exit(1)

    # 1. Hardware & Environment Telemetry
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0

    print("==================================================================")
    print("      INTELLIGATE MILESTONE 3B — YOLOv8n PLATE DETECTOR TRAINING")
    print("==================================================================")
    print(f"Device: {device_name} (CUDA: {cuda_avail})")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Initialization Weights: {init_weights}")
    print(f"Dataset Config: {data_yaml}")

    if cuda_avail:
        torch.cuda.reset_peak_memory_stats(0)

    # 2. Instantiate YOLO
    model = YOLO(str(init_weights))

    # 3. Training Run
    # 60 epochs budget with early stopping patience=12
    epochs = 60
    batch_size = 16
    imgsz = 640
    patience = 12

    train_project = root / "runs" / "detect"
    train_name = "train_plate_yolov8n"

    print("\nStarting Training Session...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        device=0 if cuda_avail else "cpu",
        workers=4,
        project=str(train_project),
        name=train_name,
        exist_ok=True,
        seed=42,
        amp=True,
        plots=True,
        save=True,
        save_period=1,
        # Augmentations tailored for license plates in street imagery
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.05,
        degrees=5.0,
        scale=0.4,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
    )

    elapsed_time = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3) if cuda_avail else 0.0

    print("\n==================================================================")
    print("                    TRAINING SESSION COMPLETED")
    print("==================================================================")
    print(f"Total Training Duration: {elapsed_time / 60.0:.2f} minutes ({elapsed_time:.1f} s)")
    print(f"Peak GPU VRAM Allocated: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB")

    run_dir = train_project / train_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"Best checkpoint: {best_pt} (Exists: {best_pt.exists()})")
    print(f"Last checkpoint: {last_pt} (Exists: {last_pt.exists()})")

    # 4. Copy weights to models directory with distinct names (do NOT overwrite vehicle detector)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    target_best = models_dir / "plate_detector_yolov8n.pt"
    target_last = models_dir / "plate_detector_yolov8n_last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, target_best)
        shutil.copy2(best_pt, models_dir / "plate_detector_yolov8n_best.pt")
        print(f"Copied best model to: {target_best}")

    if last_pt.exists():
        shutil.copy2(last_pt, target_last)
        print(f"Copied last model to: {target_last}")

    # 5. Extract training metrics from results.csv
    csv_path = run_dir / "results.csv"
    metrics_summary = {}
    if csv_path.exists():
        import csv
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # strip column whitespace
            reader.fieldnames = [c.strip() for c in reader.fieldnames]
            rows = list(reader)
        if rows:
            last_row = rows[-1]
            # Find best row by metrics/mAP50(B)
            best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0.0) or 0.0))
            best_epoch = int(best_row.get("epoch", 1))

            metrics_summary = {
                "completed_epochs": len(rows),
                "best_epoch": best_epoch,
                "best_metrics": {
                    "precision": float(best_row.get("metrics/precision(B)", 0.0) or 0.0),
                    "recall": float(best_row.get("metrics/recall(B)", 0.0) or 0.0),
                    "mAP50": float(best_row.get("metrics/mAP50(B)", 0.0) or 0.0),
                    "mAP50_95": float(best_row.get("metrics/mAP50-95(B)", 0.0) or 0.0),
                    "val_box_loss": float(best_row.get("val/box_loss", 0.0) or 0.0),
                    "val_cls_loss": float(best_row.get("val/cls_loss", 0.0) or 0.0),
                    "val_dfl_loss": float(best_row.get("val/dfl_loss", 0.0) or 0.0),
                },
                "last_metrics": {
                    "precision": float(last_row.get("metrics/precision(B)", 0.0) or 0.0),
                    "recall": float(last_row.get("metrics/recall(B)", 0.0) or 0.0),
                    "mAP50": float(last_row.get("metrics/mAP50(B)", 0.0) or 0.0),
                    "mAP50_95": float(last_row.get("metrics/mAP50-95(B)", 0.0) or 0.0),
                }
            }
            print("\nBest Validation Metrics:")
            print(f"  Best Epoch: {metrics_summary['best_epoch']}")
            print(f"  Precision: {metrics_summary['best_metrics']['precision'] * 100:.2f}%")
            print(f"  Recall:    {metrics_summary['best_metrics']['recall'] * 100:.2f}%")
            print(f"  mAP@0.50:  {metrics_summary['best_metrics']['mAP50'] * 100:.2f}%")
            print(f"  mAP@0.50:0.95: {metrics_summary['best_metrics']['mAP50_95'] * 100:.2f}%")

    # 6. Save comprehensive benchmark log
    log_file = root / "logs" / "benchmarks" / "m03b_training_summary.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "gpu": device_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "pytorch_version": torch.__version__,
        },
        "config": {
            "base_weights": str(init_weights),
            "dataset_yaml": str(data_yaml),
            "target_epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "patience": patience,
            "optimizer": "AdamW",
            "seed": 42
        },
        "training_time_seconds": round(elapsed_time, 2),
        "results": metrics_summary,
        "saved_checkpoints": {
            "best_pt": str(target_best),
            "last_pt": str(target_last)
        }
    }
    log_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable training summary to: {log_file}")

if __name__ == "__main__":
    main()
