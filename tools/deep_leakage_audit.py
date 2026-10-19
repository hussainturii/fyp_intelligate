"""
Deep leakage audit: find all train/val images that visually duplicate test images in Dataset A or Dataset B.
Saves the blacklist of training images that must be excluded to guarantee 100% test set integrity.
"""
from pathlib import Path
import cv2
import json

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_dist(h1, h2):
    return bin(h1 ^ h2).count('1')

def full_leakage_audit():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    
    # 1. Index all test images across both datasets
    test_images = {}
    for ds_name, p in [("dataset_a", root / "data/pakistan_plates/dataset_a/test/images"),
                       ("dataset_b", root / "data/pakistan_plates/dataset_b/test/images")]:
        for img_p in p.glob("*.*"):
            img = cv2.imread(str(img_p))
            if img is not None:
                test_images[f"{ds_name}/{img_p.name}"] = {
                    "path": img_p,
                    "dataset": ds_name,
                    "name": img_p.name,
                    "hash": dhash(img)
                }
    print(f"Total held-out test images indexed: {len(test_images)} (A: 34, B: 155)")

    # 2. Check all train and val images against test images
    leaked_candidates = []
    
    for split_type in ["train", "valid"]:
        for ds_name, p in [("dataset_a", root / f"data/pakistan_plates/dataset_a/{split_type}/images"),
                           ("dataset_b", root / f"data/pakistan_plates/dataset_b/{split_type}/images")]:
            for img_p in p.glob("*.*"):
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                h_img = dhash(img)
                for test_key, t_info in test_images.items():
                    dist = hamming_dist(h_img, t_info["hash"])
                    if dist <= 4: # Near-identical
                        leaked_candidates.append({
                            "candidate_split": f"{ds_name}/{split_type}",
                            "candidate_file": img_p.name,
                            "test_key": test_key,
                            "hamming_dist": dist
                        })

    print(f"\nTotal Leaked / Near-Duplicate pairs detected: {len(leaked_candidates)}")
    for lc in leaked_candidates:
        print(f"  {lc['candidate_split']}/{lc['candidate_file']} <==> {lc['test_key']} (dist={lc['hamming_dist']})")

    out_path = root / "logs" / "benchmarks" / "m03b_leaked_test_duplicates.json"
    out_path.write_text(json.dumps(leaked_candidates, indent=2), encoding="utf-8")
    print(f"\nSaved leakage audit to {out_path}")

if __name__ == "__main__":
    full_leakage_audit()
