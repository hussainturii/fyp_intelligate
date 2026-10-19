"""
model_integrity.py — Model Supply-Chain Integrity & Verification
================================================================
Validates local model weight files against the trusted SHA-256 manifest.
Ensures zero unapproved external network downloads occur during startup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models" / "trusted_manifest.json"

# Prevent Ultralytics from initiating automatic downloads over HTTP
os.environ["ULTRALYTICS_AUTO_DOWNLOAD"] = "0"
os.environ["YOLO_VERBOSE"] = "False"


def load_manifest() -> Dict[str, Any]:
    """Load the trusted model weights manifest."""
    if not MANIFEST_PATH.exists():
        logger.warning(f"[ModelIntegrity] Manifest missing at {MANIFEST_PATH}")
        return {"version": "1.0.0", "models": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupted model manifest at {MANIFEST_PATH}: {exc}") from exc


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file using chunked reads."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_model_file(
    relative_path: str,
    strict: bool = True,
    expected_sha256: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify a local model weights file against the trusted manifest or expected hash.
    
    Args:
        relative_path: Path relative to repository root (e.g. 'yolov8n.pt' or 'models/face_detection_yunet_2023mar.onnx')
        strict: If True, raises RuntimeError on missing file or hash mismatch.
        expected_sha256: Optional override for expected hash.

    Returns:
        dict with verification status, computed hash, and metadata.
    """
    p = Path(relative_path)
    full_path = p if p.is_absolute() else (ROOT / relative_path)
    manifest = load_manifest()
    model_name = p.name
    meta = manifest.get("models", {}).get(model_name, {})
    expected = expected_sha256 or meta.get("sha256")

    if not full_path.exists():
        msg = f"Required model file missing: {relative_path}. Automated network downloads are disabled for supply-chain safety."
        logger.error(f"[ModelIntegrity] {msg}")
        if strict:
            raise RuntimeError(msg)
        return {
            "name": model_name,
            "path": str(relative_path),
            "status": "MISSING",
            "verified": False,
            "error": msg
        }

    actual_hash = compute_file_sha256(full_path)
    if expected and actual_hash.lower() != expected.lower():
        msg = (
            f"Model integrity verification failed for {relative_path}!\n"
            f"Expected SHA-256: {expected}\n"
            f"Actual SHA-256:   {actual_hash}\n"
            f"The model file may be tampered, corrupted, or an unexpected version."
        )
        logger.critical(f"[ModelIntegrity] {msg}")
        if strict:
            raise RuntimeError(msg)
        return {
            "name": model_name,
            "path": str(relative_path),
            "status": "CORRUPTED",
            "verified": False,
            "expected_sha256": expected,
            "actual_sha256": actual_hash,
            "error": msg
        }

    logger.info(f"[ModelIntegrity] Verified {model_name} (SHA-256: {actual_hash[:16]}...)")
    return {
        "name": model_name,
        "path": str(relative_path),
        "status": "VERIFIED",
        "verified": True,
        "size_bytes": full_path.stat().st_size,
        "sha256": actual_hash,
        "license": meta.get("license", "Unknown"),
        "task": meta.get("task", "Unknown")
    }


def get_model_inventory() -> List[Dict[str, Any]]:
    """Return sanitized inventory and verification status for models, avoiding path exposure."""
    manifest = load_manifest()
    results = []
    for model_name, meta in manifest.get("models", {}).items():
        rel_path = meta.get("relative_path", model_name)
        status = verify_model_file(rel_path, strict=False)
        sanitized = {
            "name": model_name,
            "format": meta.get("format", "Unknown"),
            "license": meta.get("license", "Unknown"),
            "task": meta.get("task", "Unknown"),
            "source": meta.get("source", "Unknown"),
            "status": status.get("status", "UNKNOWN"),
            "verified": status.get("verified", False),
            "sha256": status.get("sha256", meta.get("sha256", "")),
            "size_bytes": status.get("size_bytes", meta.get("size_bytes", 0)),
        }
        results.append(sanitized)
    return results
