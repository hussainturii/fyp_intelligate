"""
Seed realistic, non-sensitive, academic demonstration data for IntelliGate frontend screenshots.
Populates:
- Enrolled SFace encrypted face identities
- Registered HDF5 authorized vehicle registry
- Realistic access events across face, vehicle, queue, and safety modules
- Security alerts (CRITICAL, HIGH, MEDIUM, LOW)
- Review requests (Pending operator review)
- System health metrics
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
import cv2
import numpy as np

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.vision import FaceEngine
from module2_vehicle.database.vehicle_db import VehicleDatabase


def seed():
    root = Path(__file__).resolve().parent.parent
    db_path = root / "logs" / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = EventStore(str(db_path))
    engine = FaceEngine(store)

    # 1. Enrol Realistic Faces
    print("[1/5] Enrolling demo face identities into SFace encrypted gallery...")
    lfw_dir = root / "data" / "lfw" / "lfw"
    
    # Pick a few distinct identity photos from LFW
    sample_faces = [
        ("Dr. Tariq Mahmood", "employee", "Colin_Powell/Colin_Powell_0001.jpg"),
        ("Dr. Sarah Qureshi", "resident", "Jennifer_Aniston/Jennifer_Aniston_0001.jpg"),
        ("Engr. Hussain Raza", "security_staff", "George_W_Bush/George_W_Bush_0001.jpg"),
        ("Zainab Fatima", "resident", "Serena_Williams/Serena_Williams_0001.jpg"),
    ]

    for name, role, rel_path in sample_faces:
        p = lfw_dir / rel_path
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                try:
                    engine.register(name, role, img)
                    print(f"  Enrolled face: {name} ({role})")
                except ValueError as e:
                    print(f"  Face {name} already enrolled or error: {e}")

    # 2. Register Authorized Vehicles
    print("\n[2/5] Registering authorized vehicles into HDF5 database...")
    vdb_path = root / "logs" / "cpu_vehicles.h5"
    unk_path = root / "logs" / "cpu_unknown_vehicles.h5"
    vdb = VehicleDatabase(str(vdb_path), str(unk_path))

    demo_vehicles = [
        ("LEA-1234", "Dr. Tariq Mahmood (Dean, Faculty of Computing)"),
        ("ICT-5678", "Dr. Sarah Qureshi (Chairperson, AI Department)"),
        ("MNL-8185", "Campus Security Patrol Unit 01"),
        ("LEF-3503", "Faculty Administration Transport"),
        ("AV-8907", "University Official Delegation Bus"),
    ]

    for plate, owner in demo_vehicles:
        clean_plate = ''.join(c for c in plate.upper() if c.isalnum())
        if clean_plate not in vdb._plate_map:
            vdb.register(clean_plate, owner)
            print(f"  Registered vehicle: {clean_plate} -> {owner}")
        else:
            print(f"  Vehicle {clean_plate} already registered.")

    # 3. Seed Realistic Access Events
    print("\n[3/5] Seeding access events...")
    with store._conn() as conn:
        # Clear existing simulated events if any to make clean audit trail
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM review_requests")
        conn.execute("DELETE FROM visitor_visits")

    now = time.time()
    events_data = [
        # (offset_secs_ago, module, event_type, identity, confidence, method, camera_id, meta)
        (3600 * 4.5, "vehicle", "authorized", "LEA-1234", 0.96, "exact_plate", 1, 
         {"decision": "ALLOW", "reason": "Registered exact plate with high OCR confidence", "lane": "Lane 1 North"}),
        (3600 * 4.2, "gate", "open", "access-policy", 1.0, "virtual_gate", 0,
         {"command": "OPEN", "state": "OPEN", "reason": "ALLOW for LEA-1234"}),
        (3600 * 4.1, "gate", "close", "system-timer", 1.0, "virtual_gate", 0,
         {"command": "CLOSE", "state": "CLOSED", "reason": "Auto-close after vehicle transit"}),
        (3600 * 3.8, "face", "recognition_candidate", "Dr. Sarah Qureshi", 0.94, "face_recognition", 2,
         {"decision": "REVIEW_REQUIRED", "reason": "SFace candidate match requires visual confirmation", "pedestal": "Turnstile A"}),
        (3600 * 3.7, "gate", "approved", "sec_admin", 1.0, "manual_override", 0,
         {"review_id": "rev-101", "decision": "ALLOW", "actor": "sec_admin"}),
        (3600 * 3.2, "vehicle", "stranger", "KHI-9921", 0.72, "plate", 1,
         {"decision": "DENY", "reason": "Unregistered vehicle plate", "lane": "Lane 1 North"}),
        (3600 * 2.8, "queue", "congestion", "Queue_Lane_1", 0.88, "object_tracking", 1,
         {"count": 5, "threshold": 4, "state": "CONGESTION_WARNING"}),
        (3600 * 2.4, "face", "stranger", "Visitor_Unknown_409", 0.38, "face_recognition", 2,
         {"decision": "REVIEW_REQUIRED", "reason": "Unregistered identity at pedestrian gate"}),
        (3600 * 2.1, "safety", "obstruction", "IR_Beam_01", 1.0, "hardware_interlock", 0,
         {"sensor": "safety_curtain", "state": "STOPPED", "reason": "Pedestrian crossed safety envelope during gate motion"}),
        (3600 * 2.0, "gate", "stop", "safety-interlock", 1.0, "virtual_gate", 0,
         {"command": "STOP", "state": "STOPPED", "reason": "Safety beam interruption"}),
        (3600 * 1.9, "safety", "state_changed", "IR_Beam_01", 1.0, "hardware_interlock", 0,
         {"sensor": "safety_curtain", "state": "CLEAR", "reason": "Obstruction envelope cleared"}),
        (3600 * 1.8, "gate", "reset", "sec_staff", 1.0, "virtual_gate", 0,
         {"command": "RESET", "state": "CLOSED", "reason": "Operator cleared obstruction interlock"}),
        (3600 * 1.2, "vehicle", "authorized", "ICT-5678", 0.98, "exact_plate", 1,
         {"decision": "ALLOW", "reason": "Registered faculty vehicle recognized", "lane": "Lane 1 North"}),
        (3600 * 0.9, "face", "recognition_candidate", "Engr. Hussain Raza", 0.97, "face_recognition", 2,
         {"decision": "REVIEW_REQUIRED", "reason": "Biometric staff candidate", "pedestal": "Turnstile B"}),
        (3600 * 0.5, "face", "spoof_detected", "Impersonator_Suspicious", 0.91, "liveness_check", 2,
         {"decision": "DENY", "reason": "Presentation attack detected: 2D printed photo texture anomaly"}),
        (3600 * 0.2, "vehicle", "authorized", "MNL-8185", 0.95, "exact_plate", 1,
         {"decision": "ALLOW", "reason": "Security patrol vehicle entry verified", "lane": "Main Barrier"}),
    ]

    with store._conn() as conn:
        for offset, mod, ev_type, ident, conf, meth, cam_id, meta in events_data:
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - offset))
            conn.execute(
                """INSERT INTO events (timestamp, module, event_type, identity, confidence, method, camera_id, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (t_str, mod, ev_type, ident, conf, meth, cam_id, json.dumps(meta))
            )
    print(f"  Inserted {len(events_data)} realistic access events.")

    # 4. Seed Realistic Security Alerts
    print("\n[4/5] Seeding security alerts...")
    alerts = [
        ("CRITICAL", "safety", "Safety Interlock: Virtual Gate Stopped by Obstruction",
         "Optical beam sensor triggered during downward gate motion cycle. Motion arrested within 80ms.", 0, 3600 * 2.1),
        ("HIGH", "spoof_attempt", "Biometric Liveness Rejection (Presentation Attack)",
         "Turnstile 2 YuNet/Liveness rejected high-glare presentation artifact. Identity: Impersonator_Suspicious.", 0, 3600 * 0.5),
        ("MEDIUM", "unauthorized_entry", "Unregistered Vehicle Collision Attempt at Lane 1",
         "Plate KHI-9921 not found in authorized campus registry. Virtual barrier held closed.", 0, 3600 * 3.2),
        ("MEDIUM", "repeat_visitor", "Recurrent Unregistered Visitor Appearance (3 visits in 24h)",
         "Temporal tracking identified visitor hash 7f8a91b2c4 at pedestrian turnstiles 3 times today without active visitor badge.", 0, 3600 * 1.5),
        ("LOW", "congestion", "Traffic Queue Advisory: Lane 1 North Approach",
         "Sustained vehicle queue count (5 vehicles) exceeded normal flow threshold for 180 seconds.", 0, 3600 * 2.8),
        ("LOW", "system", "Biometric Key Rotation & Model Verification Passed",
         "AES-256 GCM key verification completed; YuNet and SFace SHA-256 hashes matched registry.", 1, 3600 * 6.0),
    ]

    with store._conn() as conn:
        for sev, cat, title, desc, res, offset in alerts:
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - offset))
            conn.execute(
                """INSERT INTO alerts (timestamp, severity, category, title, description, resolved, resolved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (t_str, sev, cat, title, desc, res, t_str if res else None)
            )
    print(f"  Inserted {len(alerts)} security alerts.")

    # 5. Seed Review Requests and Visitor Tracking
    print("\n[5/5] Seeding review requests and stranger appearance tracking...")
    with store._conn() as db:
        # Pending review
        rev_id = str(uuid.uuid4())
        db.execute(
            """INSERT INTO review_requests (id, event_id, identity, status, created, expires, actor)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rev_id, 14, "Dr. Sarah Qureshi", "PENDING", now - 120, now + 600, "system")
        )
        # Stranger visits
        db.execute(
            """INSERT OR REPLACE INTO visitor_visits (id, identity, source, session, seen)
               VALUES (1, 'stranger_7f8a91b2', 'camera_2', 'sess_01', ?)""",
            (now - 1800,)
        )
        db.execute(
            """INSERT OR REPLACE INTO visitor_visits (id, identity, source, session, seen)
               VALUES (2, 'stranger_e4d2a109', 'camera_2', 'sess_02', ?)""",
            (now - 3600,)
        )

        # Health snapshots
        for i in range(15):
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - (15 - i) * 60))
            cpu_val = 14.5 + (i % 5) * 2.1
            mem_val = 42.0 + (i % 3) * 1.5
            fps_val = 28.5 + (i % 4) * 0.8
            db.execute(
                """INSERT INTO system_health (timestamp, cpu_pct, mem_pct, gpu_pct, fps, cam_online)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (t_str, cpu_val, mem_val, 0.0, fps_val)
            )

    print("\n[SUCCESS] Demo dataset successfully seeded into IntelliGate!")


if __name__ == "__main__":
    seed()
