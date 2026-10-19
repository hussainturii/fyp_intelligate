"""
Analyze visual composition: box area distribution, aspect ratio, and resolution
comparing Dataset A and Dataset B to document differences between outdoor vehicle scenes
and close-up plate crops.
"""
from pathlib import Path
import cv2
import numpy as np
import json

def parse_box(parts):
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        xc, yc, w, h = coords
        return xc, yc, w, h
    elif len(coords) >= 6:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    return None

def analyze_dataset_characteristics(name, root_dir):
    root = Path(root_dir)
    areas = []
    aspect_ratios = []
    resolutions = []
    
    for lbl_p in root.glob("**/labels/*.txt"):
        img_p = lbl_p.parent.parent / "images" / f"{lbl_p.stem}.jpg"
        if not img_p.exists():
            continue
        lines = lbl_p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            box = parse_box(parts)
            if box:
                xc, yc, w, h = box
                areas.append(w * h)
                aspect_ratios.append(w / max(1e-5, h))

    return {
        "dataset": name,
        "sample_count": len(areas),
        "mean_box_area_pct": float(np.mean(areas) * 100) if areas else 0,
        "median_box_area_pct": float(np.median(areas) * 100) if areas else 0,
        "p10_box_area_pct": float(np.percentile(areas, 10) * 100) if areas else 0,
        "p90_box_area_pct": float(np.percentile(areas, 90) * 100) if areas else 0,
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0,
        "median_aspect_ratio": float(np.median(aspect_ratios)) if aspect_ratios else 0,
    }

if __name__ == "__main__":
    da_stats = analyze_dataset_characteristics("Dataset A", "data/pakistan_plates/dataset_a")
    db_stats = analyze_dataset_characteristics("Dataset B", "data/pakistan_plates/dataset_b")
    print(json.dumps([da_stats, db_stats], indent=2))
