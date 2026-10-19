"""Safe import and extraction script for downloaded LFW and Pakistani Plate datasets.
Adheres to strict security invariants:
- Verifies path traversal safety (no leading slash, no '..')
- Preserves original downloaded archives in Downloads
- Safely extracts into project directories
- Verifies extracted structure
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

DOWNLOADS = Path("C:/Users/user/Downloads")
PROJECT_ROOT = Path("D:/MZHR_FYP/INTELLIGATE")
LFW_DEST = PROJECT_ROOT / "data/lfw"
PAK_PLATES_DEST = PROJECT_ROOT / "data/pakistan_plates"


def safe_extract_zip(zip_path: Path, target_dir: Path):
    print(f"\n[IMPORT] Extracting ZIP: {zip_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_resolved = target_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Security: check for path traversal
            member_path = (target_dir_resolved / member.filename).resolve()
            if not str(member_path).startswith(str(target_dir_resolved)):
                raise ValueError(f"Path traversal security vulnerability in {member.filename}")
            z.extract(member, target_dir)
    print(f"[SUCCESS] Extracted {zip_path.name}")


def safe_extract_tar(tar_path: Path, target_dir: Path):
    print(f"\n[IMPORT] Extracting TAR.GZ: {tar_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_resolved = target_dir.resolve()

    with tarfile.open(tar_path, "r:*") as t:
        for member in t.getmembers():
            member_path = (target_dir_resolved / member.name).resolve()
            if not str(member_path).startswith(str(target_dir_resolved)):
                raise ValueError(f"Path traversal security vulnerability in {member.name}")
        t.extractall(target_dir)
    print(f"[SUCCESS] Extracted {tar_path.name}")


def import_all():
    print("=== INTELLIGATE DATASET IMPORT & SAFETY VALIDATION ===")
    
    # 1. Pairs file
    LFW_DEST.mkdir(parents=True, exist_ok=True)
    dest_pairs = LFW_DEST / "pairs.txt"
    src_pairs = DOWNLOADS / "pairs.txt"
    if src_pairs.exists() and (not dest_pairs.exists() or dest_pairs.stat().st_size != src_pairs.stat().st_size):
        shutil.copy2(src_pairs, dest_pairs)
        print(f"[IMPORT] Copied pairs.txt to {dest_pairs} ({dest_pairs.stat().st_size:,} bytes)")
    else:
        print(f"[EXISTS] pairs.txt ready at {dest_pairs}")

    # 2. LFW Images
    # Check if data/lfw/lfw already has the 13,233 images
    dest_lfw_images = LFW_DEST / "lfw"
    existing_lfw_count = len(list(dest_lfw_images.glob("*/*.jpg"))) if dest_lfw_images.exists() else 0
    
    if existing_lfw_count >= 13200:
        print(f"[EXISTS] LFW images already extracted ({existing_lfw_count} images in {dest_lfw_images})")
    else:
        # Check if C:\Users\user\Downloads\lfw has the images
        src_lfw_dir = DOWNLOADS / "lfw"
        src_lfw_count = len(list(src_lfw_dir.glob("*/*.jpg"))) if src_lfw_dir.exists() else 0
        
        if src_lfw_count >= 13200:
            print(f"[IMPORT] Copying pre-extracted LFW images from {src_lfw_dir} ({src_lfw_count} images)...")
            shutil.copytree(src_lfw_dir, dest_lfw_images, dirs_exist_ok=True)
            print(f"[SUCCESS] LFW directory synchronized: {len(list(dest_lfw_images.glob('*/*.jpg')))} images.")
        else:
            # Extract from lfw.tgz
            src_tgz = DOWNLOADS / "lfw.tgz"
            if src_tgz.exists():
                safe_extract_tar(src_tgz, LFW_DEST)
            else:
                raise FileNotFoundError("Could not find extracted lfw directory or lfw.tgz in Downloads!")

    # 3. Pakistani Plates Dataset A
    dataset_a_zip = DOWNLOADS / "Pakistani-Number-plates.v6i.yolov8.zip"
    dest_a = PAK_PLATES_DEST / "dataset_a"
    if dataset_a_zip.exists():
        if not (dest_a / "data.yaml").exists():
            safe_extract_zip(dataset_a_zip, dest_a)
        else:
            print(f"[EXISTS] Dataset A already extracted at {dest_a}")
    else:
        print(f"[WARNING] {dataset_a_zip} not found in Downloads!")

    # 4. Pakistani Plates Dataset B
    dataset_b_zip = DOWNLOADS / "PAKISTANI_LICENSE_NUMBER_PLATES.v1i.yolov8.zip"
    dest_b = PAK_PLATES_DEST / "dataset_b"
    if dataset_b_zip.exists():
        if not (dest_b / "data.yaml").exists():
            safe_extract_zip(dataset_b_zip, dest_b)
        else:
            print(f"[EXISTS] Dataset B already extracted at {dest_b}")
    else:
        print(f"[WARNING] {dataset_b_zip} not found in Downloads!")

    # 5. Final verification summary
    print("\n=== EXTRACTION VERIFICATION SUMMARY ===")
    lfw_imgs = len(list((LFW_DEST / "lfw").glob("*/*.jpg")))
    print(f"LFW Face Images: {lfw_imgs:,} images across {len(list((LFW_DEST / 'lfw').iterdir()))} identities")
    print(f"LFW Pairs File: {dest_pairs} -> {dest_pairs.exists()} ({dest_pairs.stat().st_size:,} bytes)")
    
    a_imgs = len(list(dest_a.glob("*/*/*.jpg"))) + len(list(dest_a.glob("*/*/*.png")))
    a_lbls = len(list(dest_a.glob("*/*/*.txt")))
    print(f"Dataset A: {a_imgs} images, {a_lbls} label files (train/valid/test)")
    
    b_imgs = len(list(dest_b.glob("*/*/*.jpg"))) + len(list(dest_b.glob("*/*/*.png")))
    b_lbls = len(list(dest_b.glob("*/*/*.txt")))
    print(f"Dataset B: {b_imgs} images, {b_lbls} label files (train/valid/test)")


if __name__ == "__main__":
    import_all()
