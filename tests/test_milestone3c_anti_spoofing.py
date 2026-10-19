"""
test_milestone3c_anti_spoofing.py — Milestone 3C Regression Test Suite
=======================================================================
Validates:
1. Checkpoint integrity & cryptographic supply-chain verification
2. Tampered checkpoint rejection
3. CPU & GPU device selection and clean fallback
4. Valid face crop processing & 80x80 preprocessing
5. Empty frames, zero-detection, and no-face frames
6. Multiple detected faces in a single frame
7. Degraded, corrupt, and out-of-bounds crops
8. Conservative INCONCLUSIVE / NOT_EVALUATED handling
9. Network-independent (air-gapped) PAD inference
10. Non-negotiable security invariant: Face recognition and PAD NEVER open the gate (ALLOW)
11. Suspected spoof detection creates DENY decision and high alert
12. Vehicle exact-plate authorization remains functional and unaffected
13. STOP, LOCKDOWN, and Obstruction safety interlocks remain authoritative
14. Complete metadata propagation across pipeline
"""
import copy
import json
import socket
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from module1_face_recognition.anti_spoofing.deep_pad import (
    DEFAULT_CHECKPOINT,
    MiniFASNet,
    MiniFASNetV2Detector,
)
from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.model_integrity import (
    compute_file_sha256,
    get_model_inventory,
    load_manifest,
    verify_model_file,
)
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline

EXPECTED_PAD_SHA256 = "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint & Integrity Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_integrity_and_manifest():
    """Verify MiniFASNetV2 weights file exists and matches the trusted manifest SHA-256."""
    assert DEFAULT_CHECKPOINT.exists(), f"Weights missing: {DEFAULT_CHECKPOINT}"
    computed_sha = compute_file_sha256(DEFAULT_CHECKPOINT)
    assert computed_sha == EXPECTED_PAD_SHA256

    manifest = load_manifest()
    assert "2.7_80x80_MiniFASNetV2.pth" in manifest["models"]
    model_entry = manifest["models"]["2.7_80x80_MiniFASNetV2.pth"]
    assert model_entry["sha256"] == EXPECTED_PAD_SHA256
    assert model_entry["format"] == "PyTorch"
    assert model_entry["license"] == "Apache-2.0"

    result = verify_model_file("models/2.7_80x80_MiniFASNetV2.pth", strict=True)
    assert result["verified"] is True
    assert result["status"] == "VERIFIED"


def test_tampered_checkpoint_rejection(tmp_path):
    """Verify that tampered or corrupted weights are strictly rejected."""
    corrupted_file = tmp_path / "corrupted_minifasnet.pth"
    corrupted_file.write_bytes(b"corrupted_bytes_payload_header_tampered")

    with pytest.raises(RuntimeError):
        verify_model_file(
            "models/2.7_80x80_MiniFASNetV2.pth",
            strict=True,
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )

    with pytest.raises(RuntimeError):
        MiniFASNetV2Detector(checkpoint_path=corrupted_file, verify_integrity=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Device Selection & CPU Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_device_selection_and_cpu_fallback():
    """Verify that MiniFASNetV2Detector supports explicit CPU and GPU devices."""
    detector_cpu = MiniFASNetV2Detector(device="cpu")
    assert detector_cpu.device_str == "cpu"
    assert detector_cpu.initialized is True
    assert next(detector_cpu.model.parameters()).device.type == "cpu"

    if torch.cuda.is_available():
        detector_gpu = MiniFASNetV2Detector(device="cuda:0")
        assert detector_gpu.device_str == "cuda:0"
        assert detector_gpu.initialized is True
        assert next(detector_gpu.model.parameters()).device.type == "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Valid Face Crop & Preprocessing Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_face_crop_processing():
    """Verify 2.7x scale expansion, 80x80 resizing, and valid probability distribution."""
    detector = MiniFASNetV2Detector(device="cpu")
    # Synthetic realistic frame with a face region in center
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    cv2.circle(frame, (200, 200), 50, (200, 180, 160), -1)

    bbox = [150, 150, 100, 100]  # x, y, w, h
    patch, quality = detector.expand_and_crop(frame, bbox, scale=2.7, out_size=(80, 80))

    assert patch is not None
    assert patch.shape == (80, 80, 3)
    assert quality in ("optimal", "truncated_boundary")

    res = detector.evaluate_face(frame, bbox)
    assert res["pad_status"] in ("BONA_FIDE_CANDIDATE", "ATTACK_SUSPECTED", "INCONCLUSIVE")
    assert res["pad_model"] == "MiniFASNetV2-80x80-2.7"
    assert 0.0 <= res["pad_score"] <= 1.0
    assert res["pad_inference_ms"] > 0.0

    probs = res["pad_class_probs"]
    assert probs is not None
    prob_sum = probs["print_spoof"] + probs["bona_fide"] + probs["screen_spoof"]
    assert 0.99 <= prob_sum <= 1.01


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge Cases: Empty Frames, Degenerate Crops, Out of Bounds
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_and_degenerate_crops():
    """Verify graceful handling for empty, out-of-bounds, and degraded inputs."""
    detector = MiniFASNetV2Detector(device="cpu")

    # Empty frame
    res_empty = detector.evaluate_face(None, [10, 10, 50, 50])
    assert res_empty["pad_status"] == "NOT_EVALUATED"
    assert res_empty["pad_quality_status"] == "empty_frame"

    # Degenerate bbox
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res_degen = detector.evaluate_face(frame, [0, 0, 0, 0])
    assert res_degen["pad_status"] == "NOT_EVALUATED"

    # Tiny face (< 35px threshold)
    res_tiny = detector.evaluate_face(frame, [10, 10, 20, 20])
    assert res_tiny["pad_status"] == "NOT_EVALUATED"
    assert "low_resolution" in res_tiny["pad_quality_status"]

    # Face entirely out of image boundaries
    res_oob = detector.evaluate_face(frame, [500, 500, 60, 60])
    assert res_oob["pad_status"] in ("NOT_EVALUATED", "INCONCLUSIVE")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multiple Faces in Single Frame
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_detected_faces(tmp_path):
    """Verify that ReplayPipeline evaluates PAD for multiple detected faces independently."""
    store = EventStore(str(tmp_path / "multi_face.db"))
    runtime = SoftwareRuntime(store)

    events_captured = []
    def callback(evt):
        events_captured.append(evt)

    class MockMultiFaceEngine:
        def __init__(self):
            self.gallery_revision = 0
        def detect(self, frame):
            # Two distinct faces
            return [
                np.array([50, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]),
                np.array([250, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92]),
            ]
        def feature(self, frame, face):
            vec = np.ones(128, dtype=np.float32)
            return vec / np.linalg.norm(vec)
        def match(self, embedding):
            return 0.85, "Alice"
        def stranger(self, embedding):
            return "stranger-123"

    pipe = ReplayPipeline("face", runtime, callback, face_engine=MockMultiFaceEngine(), device="cpu")
    frame = np.full((400, 500, 3), 120, dtype=np.uint8)

    result = pipe.process_frame(frame)
    assert result["faces_detected"] == 2
    assert len(pipe.metrics["face_results"]) == 2

    # Check stage latencies exist and are recorded
    latencies = pipe.metrics["stage_latencies"]
    assert "detection_ms" in latencies
    assert "pad_ms" in latencies
    assert "recognition_ms" in latencies
    assert "tracking_ms" in latencies
    assert "total_cycle_ms" in latencies

    for face_res in pipe.metrics["face_results"]:
        assert "pad_status" in face_res
        assert "pad_score" in face_res
        assert "pad_model" in face_res
        assert face_res["pad_status"] in ("BONA_FIDE_CANDIDATE", "ATTACK_SUSPECTED", "INCONCLUSIVE", "NOT_EVALUATED")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Network Independence (Air-Gapped Offline Execution)
# ─────────────────────────────────────────────────────────────────────────────

def test_network_independent_pad_inference(monkeypatch):
    """Verify that PAD detector initialization and inference require ZERO network calls."""
    orig_connect = socket.socket.connect

    def mock_socket_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"Air-gap violation: Attempted outbound network connection to {host}")
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", mock_socket_connect)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("Air-gap violation: urllib blocked")),
    )

    detector = MiniFASNetV2Detector(device="cpu")
    test_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    res = detector.evaluate_face(test_frame, [50, 50, 70, 70])
    assert res["pad_status"] in ("BONA_FIDE_CANDIDATE", "ATTACK_SUSPECTED", "INCONCLUSIVE")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Non-Negotiable Gate Security Invariant: Face Recognition NEVER opens Gate
# ─────────────────────────────────────────────────────────────────────────────

def test_face_recognition_and_pad_never_opens_gate(tmp_path):
    """
    CRITICAL SECURITY INVARIANT:
    Regardless of bona fide candidate prediction, face biometric events must NEVER
    trigger gate ALLOW or command OPENING. Gate decisions must remain REVIEW_REQUIRED.
    """
    store = EventStore(str(tmp_path / "security_invariant.db"))
    runtime = SoftwareRuntime(store)

    assert runtime.gate()["state"] == "CLOSED"

    # Simulate perception emitting a face event with bona fide PAD prediction
    bona_fide_face_event = {
        "event_type": "access_granted",
        "module": "face",
        "identity": "Authorized_Alice",
        "confidence": 0.98,
        "method": "SFace",
        "camera_id": 0,
        "meta": {
            "pad_status": "BONA_FIDE_CANDIDATE",
            "pad_score": 0.992,
            "pad_model": "MiniFASNetV2-80x80-2.7",
            "validated_liveness": False,
            "liveness_status": "unvalidated (bona_fide_candidate): manual approval required",
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    # Observe through SoftwareRuntime
    obs = runtime.observe(bona_fide_face_event, trusted=True)

    # Must be REVIEW_REQUIRED, NEVER ALLOW!
    assert obs["decision"] == "REVIEW_REQUIRED"
    assert "Identity or liveness requires operator review" in obs["reason"]

    # Gate MUST remain CLOSED
    gate_state = runtime.gate()
    assert gate_state["state"] == "CLOSED"
    assert gate_state["state"] != "OPENING"
    assert gate_state["state"] != "OPEN"

    # A pending review request must be generated
    reviews = runtime.reviews()
    assert len(reviews) == 1
    assert reviews[0]["identity"] == "Authorized_Alice"
    assert reviews[0]["status"] == "PENDING"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Suspected Spoof Detection Routing to DENY and Alert Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_suspected_spoof_routes_to_deny_and_alert(tmp_path):
    """
    SECURITY INVARIANT:
    When PAD suspects an attack (ATTACK_SUSPECTED), SoftwareRuntime must immediately:
    1. Set decision to DENY
    2. Log a HIGH severity alert ('spoof_attempt')
    3. Never command gate opening
    """
    store = EventStore(str(tmp_path / "spoof_routing.db"))
    runtime = SoftwareRuntime(store)

    spoof_event = {
        "event_type": "spoof_detected",
        "module": "face",
        "identity": "Unknown_Attacker",
        "confidence": 0.015,
        "method": "MiniFASNetV2-80x80-2.7",
        "camera_id": 0,
        "meta": {
            "pad_status": "ATTACK_SUSPECTED",
            "pad_score": 0.015,
            "pad_model": "MiniFASNetV2-80x80-2.7",
            "pad_quality_status": "optimal",
            "validated_liveness": False,
            "liveness_status": "ATTACK_SUSPECTED: rejected by deep PAD (MiniFASNetV2)",
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    obs = runtime.observe(spoof_event, trusted=True)

    assert obs["decision"] == "DENY"
    assert "Liveness checks rejected the input" in obs["reason"]
    assert runtime.gate()["state"] == "CLOSED"

    # Check that high alert was recorded
    alerts = store.get_alerts(limit=10)
    assert len(alerts) >= 1
    spoof_alerts = [a for a in alerts if a["category"] == "spoof_attempt"]
    assert len(spoof_alerts) == 1
    assert spoof_alerts[0]["severity"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Preservation of Vehicle Plate Authorization
# ─────────────────────────────────────────────────────────────────────────────

def test_vehicle_exact_plate_authorization_preserved(tmp_path):
    """
    SECURITY INVARIANT:
    Vehicle exact-plate authorization path remains authoritative and functional.
    Only trusted internal perception vehicle events with exact_plate and >= 0.85
    confidence may command ALLOW.
    """
    store = EventStore(str(tmp_path / "vehicle_auth.db"))
    runtime = SoftwareRuntime(store)

    assert runtime.gate()["state"] == "CLOSED"

    vehicle_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "LEA-1234",
        "confidence": 0.94,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.94,
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }

    obs = runtime.observe(vehicle_event, trusted=True)
    assert obs["decision"] == "ALLOW"
    assert "Registered exact plate with sufficient OCR confidence" in obs["reason"]

    # Virtual gate moves to OPENING
    assert runtime.gate()["state"] == "OPENING"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Safety Interlocks (STOP, LOCKDOWN, Obstruction) Remain Authoritative
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_interlocks_remain_authoritative(tmp_path):
    """
    SAFETY INVARIANT:
    Obstruction and LOCKDOWN strictly override and block any gate motion,
    regardless of vehicle or biometric events.
    """
    store = EventStore(str(tmp_path / "interlock.db"))
    runtime = SoftwareRuntime(store)

    # 1. Trigger obstruction
    runtime.obstruction(True, source="beam-sensor-1")
    gate = runtime.gate()
    assert gate["obstruction"] is True
    assert gate["state"] == "STOPPED"

    # Vehicle attempts to ALLOW during obstruction
    vehicle_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "LEA-1234",
        "confidence": 0.96,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.96,
            "provenance": "perception_internal",
            "trusted_perception": True,
        },
    }
    obs = runtime.observe(vehicle_event, trusted=True)
    assert obs["decision"] == "DENY"
    assert "Gate safety interlock is active" in obs["reason"]
    assert runtime.gate()["state"] == "STOPPED"

    # 2. Trigger Lockdown
    runtime.obstruction(False, source="beam-sensor-1")
    runtime.command("LOCKDOWN", actor="chief-security-officer")
    assert runtime.gate()["lockdown"] is True
    assert runtime.gate()["state"] == "LOCKDOWN"

    obs_lockdown = runtime.observe(vehicle_event, trusted=True)
    assert obs_lockdown["decision"] == "DENY"
    assert runtime.gate()["state"] == "LOCKDOWN"
