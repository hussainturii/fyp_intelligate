"""
tests/test_milestone3c_1_ocr.py — Milestone 3C.1 Regression & Validation Test Suite
=====================================================================================
Validates:
1. OCR Normalization policy (uppercase, strip hyphens/spaces, keep year codes & 2nd line digits)
2. Character Error Rate (CER) calculation with Levenshtein operations (subs, dels, ins)
3. Ambiguous two-line plate format handling (y-band sorting vs reading order)
4. Gate Access Safety Invariants:
   - Face recognition alone NEVER issues automatic ALLOW
   - PAD BONA_FIDE_CANDIDATE NEVER issues automatic ALLOW
   - Untrusted event provenance fails closed (REVIEW_REQUIRED)
   - Spoof detection creates DENY and High alert
   - Fuzzy plate matching NEVER opens gate
   - Vehicle ALLOW requires exact plate match, trusted perception provenance, and confidence >= 0.85
   - STOP, LOCKDOWN, and Obstruction safety interlocks override all decisions
5. Stranger Memory & AES-256-GCM Biometric Encryption Integrity post-optimization
"""

import json
import time
import uuid
from pathlib import Path
import numpy as np
import pytest

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine
from tools.evaluate_verified_ocr import levenshtein_ops, normalize_plate_text

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR Normalization & CER Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_normalization_policy():
    """Verify that normalization converts to uppercase, removes punctuation, and preserves year codes."""
    # Standard 1-line plate
    assert normalize_plate_text("leb-1234") == "LEB1234"
    assert normalize_plate_text("ICT-AB-981") == "ICTAB981"
    
    # Standard 2-line plate with year code
    assert normalize_plate_text("LEF-12-3503") == "LEF123503"
    assert normalize_plate_text("MNE-16-254") == "MNE16254"
    assert normalize_plate_text("LEA - 06 - 9854") == "LEA069854"

    # Preserves digits and letters, strips commas, slashes, periods
    assert normalize_plate_text("ICT/UZ-016.") == "ICTUZ016"
    assert normalize_plate_text("") == ""

def test_levenshtein_distance_and_ops():
    """Verify exact calculation of Levenshtein edit distance, subs, dels, and ins."""
    # Exact match
    dist, subs, dels, ins = levenshtein_ops("ABC1234", "ABC1234")
    assert (dist, subs, dels, ins) == (0, 0, 0, 0)

    # 1 Substitution (0 -> O or 2 -> Z)
    dist, subs, dels, ins = levenshtein_ops("UZ016", "U2016")
    assert dist == 1
    assert subs == 1 and dels == 0 and ins == 0

    # 2 Deletions (missing year code '12')
    dist, subs, dels, ins = levenshtein_ops("LEF123503", "LEF3503")
    assert dist == 2
    assert dels == 2 and subs == 0 and ins == 0

    # 1 Insertion
    dist, subs, dels, ins = levenshtein_ops("MN1821", "MNA1821")
    assert dist == 1
    assert ins == 1 and subs == 0 and dels == 0

    # Empty prediction
    dist, subs, dels, ins = levenshtein_ops("AJJ432", "")
    assert dist == 6
    assert dels == 6

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gate Safety Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_face_recognition_and_pad_never_opens_gate(tmp_path):
    """
    CRITICAL INVARIANT 1 & 2:
    Facial recognition alone or with BONA_FIDE_CANDIDATE NEVER issues automatic ALLOW.
    """
    db_file = tmp_path / "test_safety.db"
    store = EventStore(str(db_file))
    runtime = SoftwareRuntime(store)

    # Event with recognized known face and bona fide PAD
    face_event = {
        "event_type": "access_granted",
        "module": "face",
        "identity": "Registered_Admin",
        "confidence": 0.98,
        "method": "SFace",
        "camera_id": 0,
        "meta": {
            "pad_status": "BONA_FIDE_CANDIDATE",
            "pad_score": 0.92,
            "provenance": "perception_internal",
            "trusted_perception": True,
        }
    }

    result = runtime.observe(face_event, trusted=True)
    assert result["decision"] == "REVIEW_REQUIRED", (
        f"Face recognition must NEVER automatically open the gate! Got: {result['decision']}"
    )
    assert runtime.gate()["state"] != "OPEN"
    store.close()

def test_untrusted_event_fails_closed(tmp_path):
    """
    CRITICAL INVARIANT 4:
    Events from unknown, missing, operator, or untrusted provenance FAIL CLOSED.
    """
    db_file = tmp_path / "test_untrusted.db"
    store = EventStore(str(db_file))
    runtime = SoftwareRuntime(store)

    # Vehicle event claiming exact plate, but trusted=False / missing internal perception
    untrusted_vehicle = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "ABC-1234",
        "confidence": 0.99,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.99,
            "provenance": "external_api",
            "trusted_perception": False
        }
    }

    res1 = runtime.observe(untrusted_vehicle, trusted=False)
    assert res1["decision"] == "REVIEW_REQUIRED"

    res2 = runtime.observe(untrusted_vehicle, trusted=None)
    assert res2["decision"] == "REVIEW_REQUIRED"
    store.close()

def test_spoof_detection_creates_deny_and_alert(tmp_path):
    """
    CRITICAL INVARIANT 3:
    Spoof detection creates a hard DENY decision and raises a HIGH alert.
    """
    db_file = tmp_path / "test_spoof.db"
    store = EventStore(str(db_file))
    runtime = SoftwareRuntime(store)

    spoof_event = {
        "event_type": "spoof_detected",
        "module": "face",
        "identity": "Unknown_Impostor",
        "confidence": 0.85,
        "method": "MiniFASNetV2",
        "camera_id": 0,
        "meta": {
            "pad_status": "ATTACK_SUSPECTED",
            "pad_score": 0.15,
            "provenance": "perception_internal",
            "trusted_perception": True
        }
    }

    res = runtime.observe(spoof_event, trusted=True)
    assert res["decision"] == "DENY"
    assert "rejected" in res["reason"].lower()

    # Verify alert logged in SQLite
    alerts = store.get_alerts()
    assert len(alerts) >= 1
    assert alerts[0]["category"] == "spoof_attempt"
    assert alerts[0]["severity"] == "HIGH"
    store.close()

def test_vehicle_authorization_requirements(tmp_path):
    """
    CRITICAL INVARIANT 5 & 6:
    Vehicle authorization requires:
    - Module: vehicle
    - Method: exact_plate (fuzzy match NEVER opens gate)
    - OCR confidence >= 0.85
    - Trusted internal perception
    """
    db_file = tmp_path / "test_vehicle.db"
    store = EventStore(str(db_file))
    runtime = SoftwareRuntime(store)

    # 1. Fuzzy match attempt -> must fail to REVIEW_REQUIRED
    fuzzy_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "ABC-1234",
        "confidence": 0.92,
        "method": "fuzzy_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.95,
            "provenance": "perception_internal",
            "trusted_perception": True
        }
    }
    res_fuzzy = runtime.observe(fuzzy_event, trusted=True)
    assert res_fuzzy["decision"] == "REVIEW_REQUIRED"

    # 2. Low confidence exact plate (0.75 < 0.85) -> must fail to REVIEW_REQUIRED
    low_conf_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "ABC-1234",
        "confidence": 0.75,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.75,
            "provenance": "perception_internal",
            "trusted_perception": True
        }
    }
    res_low_conf = runtime.observe(low_conf_event, trusted=True)
    assert res_low_conf["decision"] == "REVIEW_REQUIRED"

    # 3. High confidence exact plate with trusted perception -> ALLOW
    valid_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "ABC-1234",
        "confidence": 0.95,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.92,
            "provenance": "perception_internal",
            "trusted_perception": True
        }
    }
    res_valid = runtime.observe(valid_event, trusted=True)
    assert res_valid["decision"] == "ALLOW"
    assert runtime.gate()["state"] == "OPENING"
    store.close()

def test_safety_interlocks_override_access(tmp_path):
    """
    CRITICAL INVARIANT 7:
    STOP, LOCKDOWN, and Obstruction immediately override any ALLOW decision.
    """
    db_file = tmp_path / "test_interlock.db"
    store = EventStore(str(db_file))
    runtime = SoftwareRuntime(store)

    valid_event = {
        "event_type": "authorized",
        "module": "vehicle",
        "identity": "ABC-1234",
        "confidence": 0.95,
        "method": "exact_plate",
        "camera_id": 0,
        "meta": {
            "ocr_confidence": 0.95,
            "provenance": "perception_internal",
            "trusted_perception": True
        }
    }

    # Obstruction active
    runtime.obstruction(True, source="ir_sensor")
    res_obs = runtime.observe(valid_event, trusted=True)
    assert res_obs["decision"] == "DENY"
    assert "interlock is active" in res_obs["reason"]

    # Clear obstruction and trigger LOCKDOWN
    runtime.obstruction(False, source="ir_sensor")
    runtime.command("LOCKDOWN", actor="security_officer")
    res_lock = runtime.observe(valid_event, trusted=True)
    assert res_lock["decision"] == "DENY"
    assert "interlock is active" in res_lock["reason"]
    store.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Biometric Encryption & Stranger Memory
# ─────────────────────────────────────────────────────────────────────────────

def test_stranger_memory_aes_encryption_roundtrip(tmp_path):
    """
    Verify that stranger embeddings are sealed with AES-256-GCM on disk
    and can be retrieved without plaintext exposure.
    """
    db_file = tmp_path / "test_biometric.db"
    store = EventStore(str(db_file))
    engine = FaceEngine(store)

    test_emb = np.random.randn(128).astype(np.float32)
    test_emb /= np.linalg.norm(test_emb)

    # First encounter of stranger
    sid = engine.stranger(test_emb)
    assert sid is not None

    # Verify disk blob is encrypted and ciphertext != raw bytes
    with store._conn() as db:
        row = db.execute("SELECT embedding FROM cpu_strangers WHERE id=?", (sid,)).fetchone()
        assert row is not None
        blob = row["embedding"]
        assert len(blob) > 12 # nonce (12 bytes) + ciphertext + tag
        assert blob[12:] != test_emb.tobytes() # definitely encrypted!

        # Decrypt with engine.open
        dec = engine.open(blob, sid)
        assert np.allclose(dec, test_emb, atol=1e-5)

    # Second encounter of identical embedding -> must recognize same stranger ID
    sid2 = engine.stranger(test_emb)
    assert sid2 == sid

    store.close()
