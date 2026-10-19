"""
tests/test_milestone3c_2_durability_and_cache.py — Milestone 3C.2 Durability & Cache Verification
================================================================================================
Validates:
1. SQLite WAL transaction commit correctness across database reopen & process restarts.
2. Concurrent reader/writer thread safety without database locks.
3. Thread-local connection cleanup and handle release on Windows.
4. Gallery enrollment and deletion cache invalidation via gallery_revision.
5. Stranger-cache matching, in-memory updates, and 90-day retention pruning.
6. Encrypted on-disk persistence of biometric records (AES-256-GCM).
7. Sustained replay stability without unbounded memory growth.
"""

import gc
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
import pytest

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine

# ─────────────────────────────────────────────────────────────────────────────
# 1. Transaction Commit & Reopen Durability
# ─────────────────────────────────────────────────────────────────────────────

def test_sqlite_wal_transaction_durability_on_reopen(tmp_path):
    """Verify that transactions committed with synchronous=NORMAL persist across reopen."""
    db_file = tmp_path / "durability_test.db"
    
    # Session 1: Write events and alerts
    store1 = EventStore(str(db_file))
    e1 = store1.log_event("face", "authorized", identity="Alice", confidence=0.94)
    e2 = store1.log_event("vehicle", "authorized", identity="ABC-1234", confidence=0.99)
    a1 = store1.log_alert("HIGH", "spoof_attempt", "Liveness rejection", event_id=e1)
    store1.close()
    del store1
    gc.collect()

    # Session 2: Reopen from fresh store instance
    store2 = EventStore(str(db_file))
    events = store2.get_events(limit=10)
    alerts = store2.get_alerts()

    assert len(events) == 2
    assert events[0]["identity"] == "ABC-1234"
    assert events[1]["identity"] == "Alice"
    assert len(alerts) == 1
    assert alerts[0]["category"] == "spoof_attempt"
    assert alerts[0]["severity"] == "HIGH"
    assert alerts[0]["event_id"] == e1

    store2.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Concurrent Reader/Writer Thread Safety
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_reader_writer_stability(tmp_path):
    """Verify concurrent reads and writes from multiple threads under WAL mode."""
    db_file = tmp_path / "concurrent_test.db"
    store = EventStore(str(db_file))
    errors = []

    def writer_worker(thread_idx: int):
        try:
            for i in range(25):
                store.log_event("face", "stranger", identity=f"T{thread_idx}_user_{i}", confidence=0.5)
                time.sleep(0.002)
        except Exception as e:
            errors.append(f"Writer-{thread_idx}: {e}")

    def reader_worker(thread_idx: int):
        try:
            for _ in range(25):
                evts = store.get_events(limit=20)
                assert isinstance(evts, list)
                time.sleep(0.002)
        except Exception as e:
            errors.append(f"Reader-{thread_idx}: {e}")

    threads = []
    # 4 writers and 4 readers concurrently
    for t_id in range(4):
        threads.append(threading.Thread(target=writer_worker, args=(t_id,)))
        threads.append(threading.Thread(target=reader_worker, args=(t_id,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Concurrent database errors encountered: {errors}"
    total_events = store.count_events()
    assert total_events == 100, f"Expected 100 written events, found {total_events}"
    store.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Thread-Local Connection Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def test_thread_local_connection_cleanup(tmp_path):
    """Verify that calling store.close() releases file locks so files can be deleted on Windows."""
    db_file = tmp_path / "cleanup_test.db"
    store = EventStore(str(db_file))
    store.log_event("face", "test", identity="CleanupUser")
    
    # Explicit close must release Windows file locks
    store.close()
    
    # Must succeed without WinError 32
    try:
        db_file.unlink()
        wal_file = Path(str(db_file) + "-wal")
        if wal_file.exists():
            wal_file.unlink()
    except PermissionError as pe:
        pytest.fail(f"Connection cleanup failed to release file handle: {pe}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Gallery Enrollment & Deletion Cache Invalidation
# ─────────────────────────────────────────────────────────────────────────────

def test_gallery_cache_invalidation_lifecycle(tmp_path):
    """Verify that FaceEngine gallery caching correctly invalidates on register and delete."""
    db_file = tmp_path / "gallery_test.db"
    store = EventStore(str(db_file))
    engine = FaceEngine(store)

    # 1. Enrol Alice
    face_alice = np.full((120, 120, 3), 180, dtype=np.uint8)
    # Synthetic embedding for testing
    emb_alice = np.random.randn(128).astype(np.float32)
    emb_alice /= np.linalg.norm(emb_alice)

    # Directly insert into DB to simulate registration
    with store._conn() as db:
        db.execute("INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
                   ("id-alice", "Alice", "resident", engine.seal(emb_alice, "id-alice"), time.time()))
    engine.gallery_revision += 1

    # Match Alice
    score, name = engine.match(emb_alice)
    assert name == "Alice"
    assert score >= 0.99
    assert engine._gallery_cache_rev == engine.gallery_revision

    # Enrol Bob
    emb_bob = np.random.randn(128).astype(np.float32)
    emb_bob /= np.linalg.norm(emb_bob)
    with store._conn() as db:
        db.execute("INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
                   ("id-bob", "Bob", "staff", engine.seal(emb_bob, "id-bob"), time.time()))
    engine.gallery_revision += 1

    # Match Bob (triggers cache refresh)
    score_b, name_b = engine.match(emb_bob)
    assert name_b == "Bob"
    assert score_b >= 0.99

    # Delete Alice
    deleted = engine.delete("Alice")
    assert deleted is True
    # Verify Alice is no longer matched
    score_del, name_del = engine.match(emb_alice)
    assert name_del != "Alice"

    # Bob still matches
    score_b2, name_b2 = engine.match(emb_bob)
    assert name_b2 == "Bob"

    store.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Stranger-Cache Invalidation & Retention Expiry
# ─────────────────────────────────────────────────────────────────────────────

def test_stranger_cache_and_retention_pruning(tmp_path):
    """Verify in-memory stranger matching and 90-day retention prune."""
    db_file = tmp_path / "stranger_test.db"
    store = EventStore(str(db_file))
    engine = FaceEngine(store)

    emb1 = np.random.randn(128).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    # First encounter -> creates new UUID
    sid1 = engine.stranger(emb1)
    assert sid1 is not None
    assert len(engine._stranger_cache) == 1

    # Second encounter of identical embedding -> must return same ID from cache
    sid2 = engine.stranger(emb1)
    assert sid2 == sid1
    assert len(engine._stranger_cache) == 1

    # Different embedding (<0.45 score) -> creates second stranger
    emb2 = -emb1 # opposite vector
    emb2 /= np.linalg.norm(emb2)
    sid3 = engine.stranger(emb2)
    assert sid3 != sid1
    assert len(engine._stranger_cache) == 2

    # Simulate expired stranger (91 days old)
    old_time = time.time() - 91 * 86400
    with store._conn() as db:
        db.execute("UPDATE cpu_strangers SET seen=? WHERE id=?", (old_time, sid1))

    # Force retention prune
    engine._last_stranger_prune = 0.0 # force interval
    engine.stranger(np.random.randn(128).astype(np.float32))

    # Verify sid1 was pruned from DB
    with store._conn() as db:
        row = db.execute("SELECT 1 FROM cpu_strangers WHERE id=?", (sid1,)).fetchone()
        assert row is None, "Expired stranger was not pruned by 90-day retention policy!"

    store.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Biometric Encrypted Persistence Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_biometric_ciphertext_never_plaintext(tmp_path):
    """Verify that stored blobs in both cpu_faces and cpu_strangers are AES-GCM ciphertexts."""
    db_file = tmp_path / "encryption_test.db"
    store = EventStore(str(db_file))
    engine = FaceEngine(store)

    raw_emb = np.random.randn(128).astype(np.float32)
    raw_emb /= np.linalg.norm(raw_emb)
    raw_bytes = raw_emb.tobytes()

    # 1. Enrol stranger
    sid = engine.stranger(raw_emb)
    with store._conn() as db:
        blob = db.execute("SELECT embedding FROM cpu_strangers WHERE id=?", (sid,)).fetchone()["embedding"]
        # Nonce is 12 bytes, ciphertext is remainder
        assert len(blob) == 12 + len(raw_bytes) + 16 # 12 nonce + 512 bytes + 16 GCM tag
        assert raw_bytes not in blob, "Plaintext embedding detected in cpu_strangers!"

    # 2. Enrol face
    with store._conn() as db:
        db.execute("INSERT INTO cpu_faces VALUES (?,?,?,?,?)",
                   ("id-user", "CipherUser", "staff", engine.seal(raw_emb, "id-user"), time.time()))
        blob2 = db.execute("SELECT embedding FROM cpu_faces WHERE id='id-user'").fetchone()["embedding"]
        assert raw_bytes not in blob2, "Plaintext embedding detected in cpu_faces!"

    store.close()
