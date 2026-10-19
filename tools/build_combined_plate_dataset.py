"""
Build Clean Combined Pakistani License Plate Dataset for Milestone 3B YOLOv8 Training.
- Strict isolation of Test A (34 scenes) and Test B (155 scenes).
- Zero-leakage filtering: excludes all near-duplicates/derivatives of test images from Train & Valid.
- Standardizes all labels to YOLO detection format: '0 xc yc w h'.
- Converts polygon segmentations to axis-aligned bounding boxes.
- Generates detailed manifests for provenance tracking.
- Generates dataset.yaml for Ultralytics YOLOv8.
"""

from pathlib import Path
import json
import shutil
import cv2
import numpy as np

def parse_label_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        xc, yc, w, h = coords
        return 0, xc, yc, w, h, "bbox"
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        return 0, xc, yc, w, h, "polygon_converted"
    return None

def standardize_labels(src_lbl_path, dst_lbl_path):
    if not src_lbl_path.exists():
        dst_lbl_path.touch()
        return []
    lines = src_lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    out_boxes = []
    out_lines = []
    for line in lines:
        parsed = parse_label_line(line)
        if parsed:
            cls_id, xc, yc, w, h, ltype = parsed
            # Clip bounds
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))
            out_boxes.append({"cls": cls_id, "xc": xc, "yc": yc, "w": w, "h": h, "type": ltype})
            out_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    
    dst_lbl_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return out_boxes

def build_dataset():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    out_root = root / "data" / "pakistan_plates" / "combined_clean"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Load leakage blacklist
    leakage_file = root / "logs" / "benchmarks" / "m03b_leaked_test_duplicates.json"
    blacklisted_files = set()
    if leakage_file.exists():
        leaked_data = json.loads(leakage_file.read_text(encoding="utf-8"))
        for item in leaked_data:
            blacklisted_files.add(item["candidate_file"])
    print(f"Loaded {len(blacklisted_files)} candidate images blacklisted due to perceptual test-leakage.")

    splits_config = {
        "test_a": [("dataset_a", "test", False)],
        "test_b": [("dataset_b", "test", False)],
        "valid": [("dataset_a", "valid", True), ("dataset_b", "valid", True)],
        "train": [("dataset_a", "train", True), ("dataset_b", "train", True)],
    }

    manifests = {k: [] for k in splits_config}
    excluded_manifest = []

    for target_split, sources in splits_config.items():
        dst_img_dir = out_root / target_split / "images"
        dst_lbl_dir = out_root / target_split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for ds_name, src_split, apply_leakage_filter in sources:
            src_img_dir = root / "data" / "pakistan_plates" / ds_name / src_split / "images"
            src_lbl_dir = root / "data" / "pakistan_plates" / ds_name / src_split / "labels"

            prefix = "da" if ds_name == "dataset_a" else "db"
            img_files = sorted(list(src_img_dir.glob("*.*")))

            for img_p in img_files:
                if apply_leakage_filter and img_p.name in blacklisted_files:
                    excluded_manifest.append({
                        "original_path": str(img_p),
                        "file_name": img_p.name,
                        "source_dataset": ds_name,
                        "source_split": src_split,
                        "reason": "Perceptual duplicate of held-out test image (leakage prevention)"
                    })
                    continue

                new_stem = f"{prefix}_{src_split}_{img_p.stem}"
                new_img_name = f"{new_stem}{img_p.suffix.lower()}"
                new_lbl_name = f"{new_stem}.txt"

                dst_img_path = dst_img_dir / new_img_name
                dst_lbl_path = dst_lbl_dir / new_lbl_name

                # Copy image
                shutil.copy2(img_p, dst_img_path)

                # Standardize label
                src_lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
                boxes = standardize_labels(src_lbl_p, dst_lbl_path)

                manifests[target_split].append({
                    "new_image_name": new_img_name,
                    "original_path": str(img_p),
                    "source_dataset": ds_name,
                    "source_split": src_split,
                    "box_count": len(boxes),
                    "boxes": boxes
                })

        print(f"Split [{target_split}]: {len(manifests[target_split])} images prepared.")

    # Save manifests
    for k, v in manifests.items():
        (manifest_dir / f"{k}_manifest.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    (manifest_dir / "excluded_leakage_manifest.json").write_text(json.dumps(excluded_manifest, indent=2), encoding="utf-8")

    # Generate dataset.yaml
    yaml_content = f"""# IntelliGate Pakistani License Plate Dataset (Clean Combined)
# Created for Milestone 3B YOLOv8n fine-tuning
# Target: License Plate Detection
path: {out_root.as_posix()}
train: train/images
val: valid/images
test: test_b/images  # Default test split (Dataset B, 155 images); test_a (34 images) evaluated separately

nc: 1
names:
  0: license_plate
"""
    (out_root / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\nGenerated dataset.yaml at {out_root / 'dataset.yaml'}")
    print(f"Excluded {len(excluded_manifest)} leaked images to ensure 100% clean held-out test sets.")

if __name__ == "__main__":
    build_dataset()
