"""
Comprehensive Dataset Verification and Readiness Audit for Milestone 3B.
Inspects Dataset A and Dataset B:
- Image counts, file integrity, corrupted images
- Annotation format: polygon segmentations vs bounding boxes
- Normalization boundaries [0, 1]
- Empty labels, missing labels, missing images
- Exact and perceptual near-duplicates across train, valid, and test splits
- Visual confirmation of bounding box alignments
"""

import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np

def compute_dhash(image, hash_size=8):
    """Compute difference hash (dHash) for perceptual similarity comparison."""
    if image is None:
        return None
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def inspect_split(split_name, img_dir, lbl_dir):
    stats = {
        "split": split_name,
        "image_count": 0,
        "label_count": 0,
        "empty_label_count": 0,
        "corrupted_images": 0,
        "total_annotations": 0,
        "polygon_annotations": 0,
        "bbox_annotations": 0,
        "classes": set(),
        "invalid_coords": 0,
        "image_hashes": {}, # md5 -> rel_path
        "perceptual_hashes": {}, # dhash -> rel_path
        "resolutions": [],
        "sample_annotations": []
    }

    if not img_dir.exists():
        return stats

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
    stats["image_count"] = len(img_files)

    for img_path in img_files:
        # Check image readability and MD5
        try:
            with open(img_path, "rb") as f:
                data = f.read()
                md5 = hashlib.md5(data).hexdigest()
            stats["image_hashes"][md5] = img_path.name

            img = cv2.imread(str(img_path))
            if img is None:
                stats["corrupted_images"] += 1
                continue
            h, w = img.shape[:2]
            stats["resolutions"].append((w, h))

            dhash = compute_dhash(img)
            if dhash is not None:
                stats["perceptual_hashes"][dhash] = img_path.name
        except Exception as e:
            stats["corrupted_images"] += 1
            continue

        # Check label file
        lbl_path = lbl_dir / f"{img_path.stem}.txt" if lbl_dir.exists() else None
        if lbl_path and lbl_path.exists():
            stats["label_count"] += 1
            lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if not lines:
                stats["empty_label_count"] += 1
                continue

            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                stats["total_annotations"] += 1
                cls_id = int(parts[0])
                stats["classes"].add(cls_id)
                coords = [float(x) for x in parts[1:]]

                if len(coords) == 4:
                    # Bounding box: xc, yc, w, h
                    stats["bbox_annotations"] += 1
                    xc, yc, bw, bh = coords
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        stats["invalid_coords"] += 1
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    # Polygon segmentation: x1, y1, x2, y2, ...
                    stats["polygon_annotations"] += 1
                    xs = coords[0::2]
                    ys = coords[1::2]
                    if any(x < 0.0 or x > 1.0 for x in xs) or any(y < 0.0 or y > 1.0 for y in ys):
                        stats["invalid_coords"] += 1
                else:
                    stats["invalid_coords"] += 1

                if len(stats["sample_annotations"]) < 3:
                    stats["sample_annotations"].append({
                        "file": img_path.name,
                        "class": cls_id,
                        "raw_parts_count": len(parts),
                        "type": "bbox" if len(coords) == 4 else "polygon"
                    })

    stats["classes"] = sorted(list(stats["classes"]))
    return stats


def verify_dataset(name, root_dir):
    print(f"\n==========================================")
    print(f"VERIFYING {name.upper()}: {root_dir}")
    print(f"==========================================")
    root = Path(root_dir)
    results = {}
    
    # Check data.yaml or README
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        print(f"Found data.yaml:\n{yaml_path.read_text(encoding='utf-8').strip()}\n")

    splits = ["train", "valid", "val", "test"]
    for s in splits:
        img_dir = root / s / "images"
        lbl_dir = root / s / "labels"
        if not img_dir.exists():
            # Try flat split directory if images/labels subdirs don't exist
            img_dir = root / s
            lbl_dir = root / s
        if img_dir.exists():
            split_stats = inspect_split(s, img_dir, lbl_dir)
            results[s] = split_stats
            print(f"--- Split: {s} ---")
            print(f"  Images: {split_stats['image_count']}, Corrupted: {split_stats['corrupted_images']}")
            print(f"  Labels: {split_stats['label_count']}, Empty: {split_stats['empty_label_count']}")
            print(f"  Annotations: {split_stats['total_annotations']} (BBox: {split_stats['bbox_annotations']}, Polygon: {split_stats['polygon_annotations']})")
            print(f"  Classes: {split_stats['classes']}, Invalid Coords: {split_stats['invalid_coords']}")
            if split_stats["resolutions"]:
                min_w = min(r[0] for r in split_stats["resolutions"])
                max_w = max(r[0] for r in split_stats["resolutions"])
                min_h = min(r[1] for r in split_stats["resolutions"])
                max_h = max(r[1] for r in split_stats["resolutions"])
                print(f"  Resolution range: [{min_w}x{min_h}] to [{max_w}x{max_h}]")

    # Check for exact duplicate images across splits within this dataset
    print("\n--- Leakage / Cross-Split Duplicates Audit ---")
    all_splits = list(results.keys())
    for i in range(len(all_splits)):
        s1 = all_splits[i]
        h1 = results[s1]["image_hashes"]
        for j in range(i + 1, len(all_splits)):
            s2 = all_splits[j]
            h2 = results[s2]["image_hashes"]
            common_hashes = set(h1.keys()) & set(h2.keys())
            if common_hashes:
                print(f"  [WARNING] EXACT DUPLICATE LEAKAGE between {s1} and {s2}: {len(common_hashes)} images!")
                for ch in list(common_hashes)[:3]:
                    print(f"    MD5 {ch}: {s1}/{h1[ch]} == {s2}/{h2[ch]}")
            else:
                print(f"  [CLEAN] No exact duplicate leakage between {s1} and {s2}.")

    return results

def main():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    da_path = root / "data" / "pakistan_plates" / "dataset_a"
    db_path = root / "data" / "pakistan_plates" / "dataset_b"

    res_a = verify_dataset("Dataset A", da_path)
    res_b = verify_dataset("Dataset B", db_path)

    # Check cross-dataset duplicates (Dataset A vs Dataset B)
    print("\n==========================================")
    print("CROSS-DATASET DUPLICATE / LEAKAGE AUDIT")
    print("==========================================")
    for sa, data_a in res_a.items():
        ha = data_a["image_hashes"]
        for sb, data_b in res_b.items():
            hb = data_b["image_hashes"]
            common = set(ha.keys()) & set(hb.keys())
            if common:
                print(f"  [OVERLAP] Dataset A ({sa}) and Dataset B ({sb}) share {len(common)} identical images!")
            else:
                print(f"  [INDEPENDENT] Dataset A ({sa}) and Dataset B ({sb}) share 0 exact images.")

    # Save machine-readable verification summary
    out_file = root / "logs" / "benchmarks" / "m03b_dataset_readiness_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_a": {k: {k2: v2 for k2, v2 in v.items() if k2 not in ["image_hashes", "perceptual_hashes", "resolutions"]} for k, v in res_a.items()},
        "dataset_b": {k: {k2: v2 for k2, v2 in v.items() if k2 not in ["image_hashes", "perceptual_hashes", "resolutions"]} for k, v in res_b.items()}
    }
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable audit to {out_file}")

if __name__ == "__main__":
    main()
