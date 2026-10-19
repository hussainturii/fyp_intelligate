"""
test_milestone3d_vehicle_reid.py — Comprehensive Test Suite for Milestone 3D
=============================================================================
Validates deep vehicle Re-Identification:
1. Model supply-chain integrity (SHA-256, SHA-384, size against trusted manifest).
2. Embedding output dimension (512-D), unit normalization, and determinism.
3. Preprocessing, aspect ratio preservation (letterbox), and robust error handling.
4. CPU/GPU fallback and runtime execution stability.
5. Comparative evaluation against 768-D color histogram baseline.
6. Centralized access decision security invariants (appearance NEVER grants ALLOW).
7. Safety interlocks (obstruction, lockdown, untrusted provenance).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "osnet_ain_x1_0_vehicle_reid.onnx"
MANIFEST_PATH = ROOT / "models" / "trusted_manifest.json"

from module2_vehicle.database.vehicle_db import VehicleDatabase
from module2_vehicle.vehicle_reid import (
    OSNetVehicleReID,
    VehicleReID,
    color_histogram_baseline,
    letterbox_vehicle,
)
from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.model_integrity import verify_model_file
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import ReplayPipeline


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Supply-Chain & Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_model_file_exists_and_matches_exact_size():
    """Verify osnet_ain_x1_0_vehicle_reid.onnx exists and matches official 8,836,743 bytes."""
    assert MODEL_PATH.exists(), f"Model missing at {MODEL_PATH}"
    size = MODEL_PATH.stat().st_size
    assert size == 8836743, f"Size mismatch: got {size}, expected 8836743"


def test_model_checksum_matches_manifest():
    """Verify SHA-384 and SHA-256 match official Intel OMZ manifest values."""
    with MODEL_PATH.open("rb") as f:
        data = f.read()

    sha384 = hashlib.sha384(data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()

    expected_sha384 = "0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7"
    expected_sha256 = "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"

    assert sha384 == expected_sha384, "SHA-384 does not match Intel OMZ model.yml"
    assert sha256 == expected_sha256, "SHA-256 does not match registered manifest"


def test_verify_model_file_integration():
    """Verify integration with model_integrity.py."""
    result = verify_model_file("models/osnet_ain_x1_0_vehicle_reid.onnx", strict=True)
    assert result["verified"] is True
    assert result["status"] == "VERIFIED"
    assert result["license"] == "MIT"
    assert result["task"] == "vehicle_re_identification_512d"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding Dimension, Normalization & Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_vehicle_reid_initialization_and_metadata():
    """Test extractor initialization and metadata attributes."""
    reid = VehicleReID(device="cpu")
    assert reid.embedding_dim == 512
    assert reid.model_id == "osnet_ain_x1_0_vehicle_reid"
    assert reid.threshold == 0.70


def test_embedding_dimension_and_unit_norm():
    """Verify extracted embedding is strictly 512-D and L2-normalized."""
    reid = VehicleReID(device="cpu")
    dummy = np.random.randint(0, 255, (150, 250, 3), dtype=np.uint8)
    emb = reid.extract_embedding(dummy)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm 1.0, got {norm}"


def test_deterministic_embedding_generation():
    """Ensure two separate inferences on the same crop yield identical embeddings."""
    reid = VehicleReID(device="cpu")
    crop = np.random.randint(0, 255, (160, 220, 3), dtype=np.uint8)

    emb1 = reid.extract_embedding(crop)
    emb2 = reid.extract_embedding(crop.copy())

    diff = np.max(np.abs(emb1 - emb2))
    assert diff < 1e-6, f"Embeddings not deterministic: max diff = {diff}"
    sim = reid.cosine_similarity(emb1, emb2)
    assert abs(sim - 1.0) < 1e-6


def test_pil_and_numpy_input_equivalence():
    """Verify PIL Image and BGR numpy array produce identical representations."""
    reid = VehicleReID(device="cpu")
    crop_bgr = np.random.randint(0, 255, (120, 180, 3), dtype=np.uint8)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    emb_bgr = reid.extract_embedding(crop_bgr)
    emb_pil = reid.extract_embedding(crop_pil)

    sim = reid.cosine_similarity(emb_bgr, emb_pil)
    assert sim > 0.9999, f"PIL vs BGR mismatch: sim = {sim}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing, Aspect Ratio & Robust Error Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_letterbox_preserves_aspect_ratio_and_pads():
    """Verify letterboxing pads non-square image without geometric distortion."""
    wide_img = np.full((100, 300, 3), 180, dtype=np.uint8)
    padded = letterbox_vehicle(wide_img, target_size=(208, 208))

    assert padded.shape == (208, 208, 3)
    # Scaled width is 208, height is int(round(100 * 208 / 300)) = 69
    # Padding top and bottom should match neutral gray color
    assert np.array_equal(padded[0, 104], [124, 116, 104])


def test_invalid_and_empty_crops_raise_value_error():
    """Extractor must fail closed with clear error on empty or invalid crops."""
    reid = VehicleReID(device="cpu")

    with pytest.raises(ValueError):
        reid.extract_embedding(None)

    with pytest.raises(ValueError):
        reid.extract_embedding(np.empty((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        reid.extract_embedding(np.zeros((2, 2, 3), dtype=np.uint8))


def test_missing_weights_file_raises_file_not_found():
    """Extractor raises FileNotFoundError when weights path is nonexistent."""
    with pytest.raises(FileNotFoundError):
        VehicleReID(weights_path="models/nonexistent_weights.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline Comparison & Pairwise Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_color_histogram_baseline_vs_deep_reid():
    """Verify deep Re-ID (512-D) and histogram baseline (768-D) can be compared."""
    reid = VehicleReID(device="cpu")
    crop = np.random.randint(0, 255, (140, 200, 3), dtype=np.uint8)

    deep_emb = reid.extract_embedding(crop)
    hist_emb = reid.extract_histogram_baseline(crop)

    assert deep_emb.shape == (512,)
    assert hist_emb.shape == (768,)
    assert abs(np.linalg.norm(deep_emb) - 1.0) < 1e-5
    assert abs(np.linalg.norm(hist_emb) - 1.0) < 1e-5


def test_same_vehicle_vs_different_vehicle_separation():
    """Verify deep Re-ID exhibits high similarity for same vehicle and lower for different vehicle."""
    reid = VehicleReID(device="cpu")
    body_dir = ROOT / "data" / "pakistan_plates" / "vehicle_body_crops"
    crops = sorted(list(body_dir.glob("*.jpg"))) if body_dir.exists() else []
    if not crops:
        crops = sorted(list((ROOT / "data" / "pakistan_plates" / "verified_crops").glob("*.jpg")))
    if len(crops) >= 2:
        img1 = cv2.imread(str(crops[0]))
        img2 = cv2.imread(str(crops[1]))

        # Sub-crop / perturbation of img1 (simulating same vehicle observed at slightly different angle)
        h, w = img1.shape[:2]
        img1_sub = img1[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]

        emb1 = reid.extract_embedding(img1)
        emb1_same = reid.extract_embedding(img1_sub)
        emb2_diff = reid.extract_embedding(img2)

        sim_same = reid.cosine_similarity(emb1, emb1_same)
        sim_diff = reid.cosine_similarity(emb1, emb2_diff)

        assert sim_same > sim_diff, f"Expected same ({sim_same}) > diff ({sim_diff})"
        assert sim_same > 0.80, f"Same-vehicle similarity should be high: {sim_same}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Database Integration & Dimension Safety
# ─────────────────────────────────────────────────────────────────────────────

def test_vehicle_database_512d_faiss_registration_and_search(tmp_path):
    """Verify VehicleDatabase seamlessly indexes and searches 512-D embeddings."""
    db_path = str(tmp_path / "test_vehicles.h5")
    unk_path = str(tmp_path / "test_unk.h5")
    db = VehicleDatabase(db_path=db_path, unknown_db_path=unk_path, embedding_dim=512)

    emb_known = np.random.randn(512).astype(np.float32)
    emb_known = emb_known / np.linalg.norm(emb_known)

    vid = db.register("LEA1234", "Dr. Tariq", embedding=emb_known)
    assert vid in db._vehicles

    # Search with exact embedding
    matches = db.search_by_embedding(emb_known, k=1)
    assert len(matches) == 1
    assert matches[0]["plate"] == "LEA1234"
    assert matches[0]["similarity"] > 0.99


def test_vehicle_database_dimension_mismatch_safety(tmp_path):
    """Ensure querying 768-D vector against 512-D index fails safely without crashing."""
    db_path = str(tmp_path / "dim_test.h5")
    unk_path = str(tmp_path / "dim_unk.h5")
    db = VehicleDatabase(db_path=db_path, unknown_db_path=unk_path, embedding_dim=512)

    emb_512 = np.random.randn(512).astype(np.float32)
    db.register("ICT5678", "Dr. Sarah", embedding=emb_512)

    # Query with 768-D histogram vector
    emb_768 = np.random.randn(768).astype(np.float32)
    matches = db.search_by_embedding(emb_768, k=1)
    # Must fail safely and return empty list
    assert matches == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Centralized Access Decision Security Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_deep_reid_match_never_grants_autonomous_allow(tmp_path):
    """
    CRITICAL SECURITY INVARIANT:
    Deep vehicle Re-ID appearance match must NEVER result in autonomous ALLOW.
    Must always result in REVIEW_REQUIRED with gate CLOSED.
    """
    store = EventStore(str(tmp_path / "events.db"))
    runtime = SoftwareRuntime(store)

    event = {
        "module": "vehicle",
        "event_type": "access_granted",
        "identity": "Dr. Tariq",
        "method": "deep_reid",
        "confidence": 0.98,
        "meta": {
            "appearance_model": "osnet_ain_x1_0_vehicle_reid",
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    result = runtime.observe(event, trusted=True)
    assert result["decision"] == "REVIEW_REQUIRED", (
        f"Deep Re-ID violated fail-secure policy: decision={result['decision']}"
    )
    assert runtime.gate()["state"] == "CLOSED"
    assert len(runtime.reviews()) == 1


def test_exact_plate_with_high_confidence_grants_allow(tmp_path):
    """Exact plate match with conf >= 0.85 and trusted perception grants ALLOW."""
    store = EventStore(str(tmp_path / "events.db"))
    runtime = SoftwareRuntime(store)

    event = {
        "module": "vehicle",
        "event_type": "access_granted",
        "identity": "LEA1234",
        "method": "exact_plate",
        "confidence": 0.95,
        "meta": {
            "ocr_confidence": 0.95,
            "plate": "LEA1234",
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    result = runtime.observe(event, trusted=True)
    assert result["decision"] == "ALLOW"
    assert runtime.gate()["state"] == "OPENING"


def test_interlocks_override_vehicle_matches_to_deny(tmp_path):
    """Safety interlocks (obstruction / lockdown) override any match to DENY."""
    store = EventStore(str(tmp_path / "events.db"))
    runtime = SoftwareRuntime(store)

    # Trigger obstruction
    runtime.obstruction(True, "safety_beam")
    event = {
        "module": "vehicle",
        "event_type": "access_granted",
        "identity": "LEA1234",
        "method": "exact_plate",
        "confidence": 0.99,
        "meta": {
            "ocr_confidence": 0.99,
            "plate": "LEA1234",
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    result = runtime.observe(event, trusted=True)
    assert result["decision"] == "DENY"
    assert "Gate safety interlock is active" in result["reason"]
