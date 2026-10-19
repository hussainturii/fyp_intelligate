import csv
import json
import hashlib
from pathlib import Path

def main():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    res_path = root / "runs/detect/train_plate_yolov8n/results.csv"
    
    with open(res_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        rows = list(reader)
        
    print(f"Total Epochs in results.csv: {len(rows)}")
    
    # Find best epoch by mAP50 and by mAP50-95
    best_map50_row = max(rows, key=lambda x: float(x["metrics/mAP50(B)"]))
    best_map50_95_row = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
    
    print("\nBest by mAP50:")
    print(f"  Epoch: {best_map50_row['epoch']}")
    print(f"  P: {float(best_map50_row['metrics/precision(B)']):.4f}")
    print(f"  R: {float(best_map50_row['metrics/recall(B)']):.4f}")
    print(f"  mAP50: {float(best_map50_row['metrics/mAP50(B)']):.4f}")
    print(f"  mAP50-95: {float(best_map50_row['metrics/mAP50-95(B)']):.4f}")
    
    print("\nBest by mAP50-95:")
    print(f"  Epoch: {best_map50_95_row['epoch']}")
    print(f"  P: {float(best_map50_95_row['metrics/precision(B)']):.4f}")
    print(f"  R: {float(best_map50_95_row['metrics/recall(B)']):.4f}")
    print(f"  mAP50: {float(best_map50_95_row['metrics/mAP50(B)']):.4f}")
    print(f"  mAP50-95: {float(best_map50_95_row['metrics/mAP50-95(B)']):.4f}")

    print("\nLast Epoch:")
    last_row = rows[-1]
    print(f"  Epoch: {last_row['epoch']}")
    print(f"  P: {float(last_row['metrics/precision(B)']):.4f}")
    print(f"  R: {float(last_row['metrics/recall(B)']):.4f}")
    print(f"  mAP50: {float(last_row['metrics/mAP50(B)']):.4f}")
    print(f"  mAP50-95: {float(last_row['metrics/mAP50-95(B)']):.4f}")

if __name__ == "__main__":
    main()
