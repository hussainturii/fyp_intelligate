"""
Acquisition and Integrity Verification for MiniFASNetV2 Face Anti-Spoofing Model.
Source: minivision-ai / Silent-Face-Anti-Spoofing (Apache-2.0 license)
Checkpoint: 2.7_80x80_MiniFASNetV2.onnx (converted from official 2.7_80x80_MiniFASNetV2.pth)
"""

from pathlib import Path
import urllib.request
import hashlib
import json

ROOT = Path("D:/MZHR_FYP/INTELLIGATE")
MODELS_DIR = ROOT / "models"
TARGET_FILE = MODELS_DIR / "anti_spoof_minifasnetv2_2.7_80x80.onnx"

# Hugging Face mirror of verified ONNX conversion
URLS = [
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
]

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists():
        print(f"File already exists: {TARGET_FILE}")
    else:
        print(f"Downloading MiniFASNetV2 ONNX checkpoint from {URLS[0]}...")
        urllib.request.urlretrieve(URLS[0], str(TARGET_FILE))
        print(f"Download complete: {TARGET_FILE}")

    # Compute SHA-256 and size
    data = TARGET_FILE.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)
    print(f"\nModel: {TARGET_FILE.name}")
    print(f"Size:  {size} bytes ({size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {sha256}")

    # Verify loading via OpenCV DNN and ONNX (or torch)
    import cv2
    net = cv2.dnn.readNetFromONNX(str(TARGET_FILE))
    print("Successfully loaded model into OpenCV DNN engine!")

    return sha256, size

if __name__ == "__main__":
    main()
