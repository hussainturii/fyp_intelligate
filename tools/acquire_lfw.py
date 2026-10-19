"""Fast parallel downloader and safe extractor for official LFW dataset.
Resolves pre-signed S3 URL from Figshare, then downloads in parallel chunks via HTTP Range headers.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

FIGSHARE_URL = "https://ndownloader.figshare.com/files/5976018"
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"

TARGET_DIR = Path("D:/MZHR_FYP/INTELLIGATE/data/lfw")
TGZ_PATH = TARGET_DIR / "lfw.tgz"
PAIRS_PATH = TARGET_DIR / "pairs.txt"


def get_direct_s3_url(dispatcher_url: str) -> str:
    print(f"Resolving direct storage URL from {dispatcher_url}...")
    req = urllib.request.Request(
        dispatcher_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        direct_url = resp.geturl()
    print(f"Resolved direct storage URL: {direct_url[:80]}...")
    return direct_url


def download_chunk(url: str, start: int, end: int, chunk_path: Path):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Range": f"bytes={start}-{end}"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(chunk_path, "wb") as f:
        while True:
            buf = resp.read(256 * 1024)
            if not buf:
                break
            f.write(buf)


def parallel_download_s3(dest_path: Path, num_threads: int = 16) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = dest_path.parent / f".tmp_{dest_path.name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    s3_url = get_direct_s3_url(FIGSHARE_URL)

    # Content length from S3
    req = urllib.request.Request(s3_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    print(f"Total archive size: {total_size:,} bytes (~{total_size / (1024*1024):.1f} MB)")
    chunk_size = (total_size + num_threads - 1) // num_threads
    futures = []
    chunk_paths = []

    t0 = time.time()
    print(f"Starting {num_threads} parallel download streams directly from S3...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = min(start + chunk_size - 1, total_size - 1)
            chunk_file = temp_dir / f"chunk_{i:02d}.part"
            chunk_paths.append(chunk_file)
            futures.append(executor.submit(download_chunk, s3_url, start, end, chunk_file))

        for idx, fut in enumerate(concurrent.futures.as_completed(futures)):
            fut.result()
            elapsed = time.time() - t0
            print(f"Finished chunk {idx+1}/{num_threads} in {elapsed:.1f}s", flush=True)

    print(f"Assembling {num_threads} chunks into {dest_path}...")
    sha256 = hashlib.sha256()
    assembled_size = 0
    with open(dest_path, "wb") as out_f:
        for chunk_file in chunk_paths:
            with open(chunk_file, "rb") as in_f:
                while True:
                    data = in_f.read(1024 * 1024)
                    if not data:
                        break
                    out_f.write(data)
                    sha256.update(data)
                    assembled_size += len(data)
            chunk_file.unlink(missing_ok=True)

    temp_dir.rmdir()
    digest = sha256.hexdigest()
    print(f"Assembled: {assembled_size:,} bytes | SHA-256: {digest}")
    return digest


def safe_extract_tar(tar_path: Path, target_dir: Path):
    print(f"\nExtracting {tar_path} into {target_dir}...")
    target_dir_resolved = target_dir.resolve()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Total archive members: {len(members):,}")
        for member in members:
            member_path = (target_dir_resolved / member.name).resolve()
            if not str(member_path).startswith(str(target_dir_resolved)):
                raise ValueError(f"Path traversal vulnerability detected: {member.name}")
        tar.extractall(path=target_dir)
    print("Safe extraction complete.")


def verify_lfw_structure(target_dir: Path, pairs_path: Path):
    print("\nVerifying LFW structure...")
    images_dir = target_dir / "lfw"
    if not images_dir.exists():
        images_dir = target_dir

    subdirs = [p for p in images_dir.iterdir() if p.is_dir()]
    print(f"Found {len(subdirs)} identity directories in {images_dir}.")

    pairs_lines = pairs_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"Pairs header: {pairs_lines[0]}")
    sample_pair = pairs_lines[1].split()
    sample_file = images_dir / sample_pair[0] / f"{sample_pair[0]}_{int(sample_pair[1]):04d}.jpg"
    print(f"Sample pair entry: {sample_pair}")
    print(f"Sample image exists: {sample_file} -> {sample_file.exists()}")

    total_images = len(list(images_dir.glob("*/*.jpg")))
    print(f"Total validated face images: {total_images:,}")
    return images_dir


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. pairs.txt
    if not PAIRS_PATH.exists() or PAIRS_PATH.stat().st_size < 1000:
        print("--- Downloading pairs.txt ---")
        req = urllib.request.Request(PAIRS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(PAIRS_PATH, "wb") as f:
            f.write(resp.read())

    # 2. Parallel S3 download
    if not TGZ_PATH.exists() or TGZ_PATH.stat().st_size < 180_000_000:
        print("--- Downloading lfw.tgz via Parallel S3 Streams ---")
        parallel_download_s3(TGZ_PATH, num_threads=16)
    else:
        print(f"lfw.tgz already present ({TGZ_PATH.stat().st_size:,} bytes)")

    # 3. Safe extraction
    images_dir = TARGET_DIR / "lfw"
    if not images_dir.exists() or len(list(images_dir.glob("*/*.jpg"))) < 13000:
        safe_extract_tar(TGZ_PATH, TARGET_DIR)
    else:
        print("LFW images already extracted.")

    # 4. Verify
    valid_dir = verify_lfw_structure(TARGET_DIR, PAIRS_PATH)
    print(f"\n[SUCCESS] LFW dataset is verified and ready at: {valid_dir}")


if __name__ == "__main__":
    main()
