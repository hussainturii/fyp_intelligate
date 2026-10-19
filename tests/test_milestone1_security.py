"""
tests/test_milestone1_security.py — Milestone 1 Security & Provenance Tests
===========================================================================
Targeted tests for:
1. Key availability and recovery.
2. Incorrect biometric key detection.
3. Missing key with existing records (fail-secure data preservation).
4. Event provenance spoofing prevention.
5. Unauthorized and viewer event submission rejection.
6. Trusted internal event ingestion.
7. Missing model weights handling.
8. Corrupted model weights rejection.
9. Offline model inventory and verification API.
10. Gate safety invariants preservation.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.key_manager import (
    ENV_KEY_VAR,
    ENV_KEY_FILE_VAR,
    resolve_biometric_key,
)
from module8_dashboard.backend.main import app, on_pipeline_event
from module8_dashboard.backend.model_integrity import (
    compute_file_sha256,
    get_model_inventory,
    verify_model_file,
)
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ROOT


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """Provides a TestClient initialized with an admin and a security_staff account."""
    import importlib
    monkeypatch.setenv("IG_EVENT_DB", str(tmp_path / "api_security.db"))
    import module8_dashboard.backend.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        credentials = {"username": "sec_admin", "password": "admin-password-123"}
        assert client.post("/api/auth/setup", json=credentials).status_code == 200
        assert client.post("/api/auth/login", json=credentials).status_code == 200
        assert client.post("/api/auth/users", json={"username": "sec_staff", "password": "staff-password-123", "role": "security_staff"}).status_code == 200
        assert client.post("/api/auth/users", json={"username": "sec_viewer", "password": "viewer-password-123", "role": "viewer"}).status_code == 200
        yield client


# ── 1. Key Availability and Recovery ──────────────────────────────────────────

def test_biometric_key_env_injection_and_recovery(tmp_path, monkeypatch):
    """Verify key can be injected via INTELLIGATE_BIOMETRIC_KEY and recovered across reloads."""
    store = EventStore(str(tmp_path / "biometric_test.db"))
    fixed_key = AESGCM.generate_key(bit_length=256)
    monkeypatch.setenv(ENV_KEY_VAR, fixed_key.hex())

    # Initialize FaceEngine with injected key
    engine = FaceEngine(store)
    vector = np.ones(128, dtype=np.float32)
    with store._conn() as db:
        db.execute(
            "INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
            ("user-1", "Alice", "admin", engine.seal(vector, "user-1"), 1.0)
        )

    # Reload FaceEngine with same environment key
    reloaded_engine = FaceEngine(store)
    with store._conn() as db:
        row = db.execute("SELECT embedding FROM cpu_faces WHERE id='user-1'").fetchone()
    recovered_vector = reloaded_engine.open(row["embedding"], "user-1")
    assert np.allclose(recovered_vector, vector)


# ── 2. Incorrect Biometric Key Detection ──────────────────────────────────────

def test_incorrect_biometric_key_fails_safely(tmp_path, monkeypatch):
    """Verify that starting with an incorrect key raises RuntimeError and does not wipe the database."""
    store = EventStore(str(tmp_path / "biometric_wrong_key.db"))
    key_a = AESGCM.generate_key(bit_length=256)
    monkeypatch.setenv(ENV_KEY_VAR, key_a.hex())

    engine = FaceEngine(store)
    vector = np.full(128, 0.5, dtype=np.float32)
    with store._conn() as db:
        db.execute(
            "INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
            ("user-2", "Bob", "staff", engine.seal(vector, "user-2"), 1.0)
        )

    # Now attempt to start with Key B (wrong key)
    key_b = AESGCM.generate_key(bit_length=256)
    monkeypatch.setenv(ENV_KEY_VAR, key_b.hex())

    with pytest.raises(RuntimeError) as exc_info:
        FaceEngine(store)
    assert "failed authentication against existing" in str(exc_info.value).lower()

    # Verify records remain intact in database
    with store._conn() as db:
        count = db.execute("SELECT COUNT(*) FROM cpu_faces").fetchone()[0]
    assert count == 1


# ── 3. Missing Key with Existing Records Fails Safely ─────────────────────────

def test_missing_key_with_existing_records_refuses_regeneration(tmp_path, monkeypatch):
    """Verify that if key is lost while records exist, system does NOT overwrite with a new key."""
    db_path = tmp_path / "biometric_lost.db"
    store = EventStore(str(db_path))
    default_key_path = tmp_path / "biometric.key"

    monkeypatch.delenv(ENV_KEY_VAR, raising=False)
    monkeypatch.delenv(ENV_KEY_FILE_VAR, raising=False)

    # Engine generates local biometric.key on fresh init
    engine = FaceEngine(store)
    vector = np.ones(128, dtype=np.float32)
    with store._conn() as db:
        db.execute(
            "INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
            ("user-3", "Charlie", "admin", engine.seal(vector, "user-3"), 1.0)
        )

    # Simulate deletion of biometric.key
    if default_key_path.exists():
        default_key_path.unlink()

    # Attempt restart without key
    with pytest.raises(RuntimeError) as exc_info:
        FaceEngine(store)
    assert "refusing to generate a new key" in str(exc_info.value).lower()


# ── 4. Event Provenance Spoofing Prevention ───────────────────────────────────

def test_event_provenance_spoofing_prevented(auth_client):
    """Verify client-submitted events are stamped with operator provenance and cannot grant access."""
    # Login as security staff
    auth_client.post("/api/auth/login", json={"username": "sec_staff", "password": "staff-password-123"})

    # Attempt to forge an automated face access grant with perfect liveness
    response = auth_client.post(
        "/api/events",
        json={
            "module": "face",
            "event_type": "access_granted",
            "identity": "ImpersonatedAdmin",
            "confidence": 0.99,
            "method": "face_recognition",
            "meta": {
                "validated_liveness": True,
                "liveness_score": 1.0,
                "trusted_perception": True
            }
        }
    )
    assert response.status_code == 200
    data = response.json()

    # Access MUST NOT be granted
    assert data["decision"] == "REVIEW_REQUIRED"
    assert "Operator" in data["reason"] or "review" in data["reason"].lower()

    # Check event record in database
    events = auth_client.get("/api/events?limit=1").json()
    assert len(events) >= 1
    latest = events[0]
    meta = json.loads(latest["meta"]) if isinstance(latest["meta"], str) else latest["meta"]
    assert meta["provenance"] == "operator_annotation"
    assert meta["submitted_by"] == "sec_staff"
    assert meta["trusted_perception"] is False
    assert meta["validated_liveness"] is False


# ── 5. Unauthorized and Viewer Event Submission ───────────────────────────────

def test_unauthorized_and_viewer_event_submission_rejected(auth_client):
    """Verify that unauthenticated callers get 401 and viewers get 403 on POST /api/events."""
    # Logout
    auth_client.post("/api/auth/logout")

    # Unauthenticated -> 401
    resp = auth_client.post("/api/events", json={"module": "face", "event_type": "test"})
    assert resp.status_code == 401

    # Login as viewer -> 403
    auth_client.post("/api/auth/login", json={"username": "sec_viewer", "password": "viewer-password-123"})
    resp_viewer = auth_client.post("/api/events", json={"module": "face", "event_type": "test"})
    assert resp_viewer.status_code == 403


# ── 6. Trusted Internal Event Ingestion ───────────────────────────────────────

def test_trusted_internal_event_ingestion(tmp_path, monkeypatch):
    """Verify internal pipeline callback marks events as perception_internal and trusted."""
    store = EventStore(str(tmp_path / "internal_test.db"))
    runtime = SoftwareRuntime(store)

    event = {
        "module": "vehicle",
        "event_type": "authorized",
        "identity": "LEA-1234",
        "confidence": 0.95,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {"ocr_confidence": 0.92, "trusted_perception": True, "provenance": "perception_internal"}
    }
    # An internal perception event only grants ALLOW when trusted=True is explicitly passed
    result = runtime.observe(event, trusted=True)
    assert result["decision"] == "ALLOW"
    assert result["gate"]["state"] == "OPENING"

    # Regression check: If trusted is omitted or None, even valid internal metadata MUST fail closed
    untrusted_default = runtime.observe(event)
    assert untrusted_default["decision"] == "REVIEW_REQUIRED"

    untrusted_none = runtime.observe(event, trusted=None)
    assert untrusted_none["decision"] == "REVIEW_REQUIRED"

    untrusted_explicit = runtime.observe(event, trusted=False)
    assert untrusted_explicit["decision"] == "REVIEW_REQUIRED"


# ── 7. Missing Model Weights Handling ─────────────────────────────────────────

def test_missing_model_weights_handled_cleanly():
    """Verify verify_model_file behaves correctly on missing weights without external downloads."""
    with pytest.raises(RuntimeError) as exc_info:
        verify_model_file("models/nonexistent_dummy_weight.onnx", strict=True)
    assert "missing" in str(exc_info.value).lower()
    assert "automated network downloads are disabled" in str(exc_info.value).lower()

    # Non-strict mode returns structured diagnostic dict
    diag = verify_model_file("models/nonexistent_dummy_weight.onnx", strict=False)
    assert diag["status"] == "MISSING"
    assert diag["verified"] is False


# ── 8. Corrupted Model Weights Rejection ───────────────────────────────────────

def test_corrupted_model_weights_rejected(tmp_path, monkeypatch):
    """Verify that tampered or corrupted model weights fail SHA-256 verification."""
    corrupted_file = tmp_path / "corrupted_model.pt"
    corrupted_file.write_bytes(b"tampered fake model binary content")

    with pytest.raises(RuntimeError) as exc_info:
        verify_model_file(
            corrupted_file,
            strict=True,
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000"
        )
    assert "integrity verification failed" in str(exc_info.value).lower()


# ── 9. Offline Model Inventory API ────────────────────────────────────────────

def test_offline_model_inventory_api(auth_client):
    """Verify GET /api/system/models enforces RBAC, sanitizes paths, and returns verified status."""
    # 1. Unauthenticated -> 401
    auth_client.post("/api/auth/logout")
    assert auth_client.get("/api/system/models").status_code == 401

    # 2. Viewer -> 403
    auth_client.post("/api/auth/login", json={"username": "sec_viewer", "password": "viewer-password-123"})
    assert auth_client.get("/api/system/models").status_code == 403

    # 3. Admin / Staff -> 200
    auth_client.post("/api/auth/login", json={"username": "sec_admin", "password": "admin-password-123"})
    resp = auth_client.get("/api/system/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) >= 3

    model_map = {m["name"]: m for m in models}
    assert "face_detection_yunet_2023mar.onnx" in model_map
    assert "face_recognition_sface_2021dec.onnx" in model_map
    assert "yolov8n.pt" in model_map

    for name in ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx", "yolov8n.pt"]:
        item = model_map[name]
        assert item["status"] == "VERIFIED"
        assert item["verified"] is True
        assert len(item["sha256"]) == 64
        # Assert no sensitive internal host or absolute filesystem paths are exposed
        for key, val in item.items():
            if isinstance(val, str):
                assert ":" not in val or val.startswith("http"), f"Leaked path in {key}: {val}"
                assert "\\\\" not in val
                assert ".venv" not in val


# ── 10. Gate Safety Invariants Preservation ───────────────────────────────────

def test_gate_safety_invariants_preserved(tmp_path):
    """Verify STOP, obstruction, lockdown, and restart fail-secure invariants."""
    store = EventStore(str(tmp_path / "invariants.db"))
    runtime = SoftwareRuntime(store)

    # Invariant 1: Fresh gate is CLOSED
    assert runtime.gate()["state"] == "CLOSED"

    # Invariant 2: Open command initiates OPENING
    runtime.command("OPEN", "test-operator")
    assert runtime.gate()["state"] == "OPENING"

    # Invariant 3: STOP preempts motion immediately
    runtime.command("STOP", "safety-switch")
    assert runtime.gate()["state"] == "STOPPED"

    # Invariant 4: Obstruction blocks motion commands
    runtime.obstruction(True, "beam-sensor")
    assert runtime.gate()["obstruction"] is True
    with pytest.raises(ValueError):
        runtime.command("OPEN", "test-operator")
    with pytest.raises(ValueError):
        runtime.command("CLOSE", "test-operator")

    # Invariant 5: Lockdown blocks motion
    runtime.obstruction(False, "beam-sensor")
    runtime.command("LOCKDOWN", "emergency")
    assert runtime.gate()["lockdown"] is True
    with pytest.raises(ValueError):
        runtime.command("OPEN", "test-operator")

    # Invariant 6: Restart recovery from motion state forces STOPPED
    runtime.save("gate", {"state": "OPENING", "due": 9999999999.0, "obstruction": False, "lockdown": False})
    new_runtime = SoftwareRuntime(store)
    assert new_runtime.gate()["state"] == "STOPPED"
    assert "Restart requires operator review" in new_runtime.gate()["reason"]


# ── 11. Negative Provenance Permutations ──────────────────────────────────────

def test_negative_provenance_permutations_cannot_grant_allow(tmp_path):
    """Verify that missing, false, malformed, or forged provenance CANNOT grant ALLOW."""
    store = EventStore(str(tmp_path / "provenance_neg.db"))
    runtime = SoftwareRuntime(store)

    # All permutations attempt to claim exact_plate match with 0.99 confidence
    bad_metas = [
        {},                                                 # Missing provenance entirely
        {"provenance": None},                               # None provenance
        {"provenance": "operator_annotation"},              # Operator annotation
        {"provenance": "forged_internal"},                  # Fake string
        {"provenance": 12345},                              # Malformed type
        {"provenance": ["perception_internal"]},            # Malformed list
        {"trusted_perception": False},                      # Explicit False
        {"trusted_perception": "True"},                     # String instead of bool
        {"trusted_perception": 1},                          # Int instead of bool
        {"trusted_perception": None},                       # None
        {"provenance": "perception_internal", "trusted_perception": False},  # Contradictory
        {"provenance": "perception_internal", "trusted_perception": "true"}, # String true
    ]

    for bad_meta in bad_metas:
        event = {
            "module": "vehicle",
            "event_type": "access_granted",
            "method": "exact_plate",
            "meta": {"ocr_confidence": 0.99, **bad_meta}
        }
        res = runtime.observe(event)
        assert res["decision"] != "ALLOW", f"Failed closed check on bad meta: {bad_meta}"
        assert res["decision"] == "REVIEW_REQUIRED"

    # Forged internal metadata with omitted, None, or untrusted parameter MUST fail closed
    forged_event = {
        "module": "vehicle",
        "event_type": "access_granted",
        "method": "exact_plate",
        "meta": {"ocr_confidence": 0.99, "provenance": "perception_internal", "trusted_perception": True}
    }
    # 1. Omitted trusted parameter
    assert runtime.observe(forged_event)["decision"] == "REVIEW_REQUIRED"
    # 2. Explicit trusted=None
    assert runtime.observe(forged_event, trusted=None)["decision"] == "REVIEW_REQUIRED"
    # 3. Explicit trusted=False
    assert runtime.observe(forged_event, trusted=False)["decision"] == "REVIEW_REQUIRED"
    # 4. Truthy non-boolean string or int
    assert runtime.observe(forged_event, trusted="True")["decision"] == "REVIEW_REQUIRED"
    assert runtime.observe(forged_event, trusted=1)["decision"] == "REVIEW_REQUIRED"
    # 5. ONLY explicit trusted=True grants ALLOW
    authorized_res = runtime.observe(forged_event, trusted=True)
    assert authorized_res["decision"] == "ALLOW"
    assert authorized_res["gate"]["state"] == "OPENING"


# ── 12. Network-Denied Offline Startup ────────────────────────────────────────

def test_network_denied_offline_startup(tmp_path, monkeypatch):
    """Verify that full application startup, API routing, and all model loading work 100% offline."""
    import socket
    import urllib.request
    from fastapi.testclient import TestClient
    from module8_dashboard.backend.main import app
    from module8_dashboard.backend.vision import ReplayPipeline
    from module2_vehicle.plate_detector import PlateDetector

    orig_connect = socket.socket.connect

    def mock_socket_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"Simulated air-gap: Outbound network connection to {host} is blocked")
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", mock_socket_connect)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("Simulated air-gap: urllib blocked"))
    )

    # 1. Genuine full FastAPI application startup with lifespan event loop and background worker
    with TestClient(app) as client:
        status_resp = client.get("/api/auth/status")
        assert status_resp.status_code == 200

    # 2. FaceEngine loads local YuNet and SFace ONNX models without network
    store = EventStore(str(tmp_path / "offline_startup.db"))
    engine = FaceEngine(store)
    assert engine.detector is not None
    assert engine.recognizer is not None

    # 3. Vision replay pipeline initializes without network
    runtime = SoftwareRuntime(store)
    pipe = ReplayPipeline("face", runtime, lambda e: None, face_engine=engine)
    assert pipe.active is True

    # 4. Vehicle plate detector initializes without network
    plate_detector = PlateDetector(device="cpu", allow_download=False)
    assert plate_detector is not None

    # 5. Model integrity inventory computes hashes for YuNet, SFace, YOLOv8n offline
    inventory = get_model_inventory()
    assert len(inventory) >= 3
    for m in inventory:
        assert m["verified"] is True
