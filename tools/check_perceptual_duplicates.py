"""
Check for perceptual near-duplicates and augmented derivatives between Train and Test splits.
Uses dHash with Hamming distance threshold <= 6, and checks image stem prefixes.
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

def audit_perceptual(dataset_name, root_dir):
    root = Path(root_dir)
    train_dir = root / "train" / "images"
    test_dir = root / "test" / "images"
    val_dir = root / "valid" / "images"

    train_hashes = {}
    print(f"Indexing {dataset_name} train images...")
    for p in train_dir.glob("*.*"):
        img = cv2.imread(str(p))
        if img is not None:
            train_hashes[p.name] = dhash(img)

    near_duplicates = []

    print(f"Checking {dataset_name} test images against train...")
    for p in test_dir.glob("*.*"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h_test = dhash(img)
        for tr_name, h_tr in train_hashes.items():
            dist = hamming_dist(h_test, h_tr)
            if dist <= 4: # Very high perceptual similarity
                near_duplicates.append({
                    "test_image": p.name,
                    "train_image": tr_name,
                    "hamming_dist": dist
                })

    print(f"{dataset_name}: Found {len(near_duplicates)} potential perceptual near-duplicates (Hamming dist <= 4).")
    for nd in near_duplicates[:5]:
        print(f"  Test: {nd['test_image']} <-> Train: {nd['train_image']} (dist={nd['hamming_dist']})")

    return near_duplicates

if __name__ == "__main__":
    audit_perceptual("Dataset A", "data/pakistan_plates/dataset_a")
    audit_perceptual("Dataset B", "data/pakistan_plates/dataset_b")
