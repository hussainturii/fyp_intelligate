"""
profile_face_runtime.py — Active Face Perception Runtime Profiler
==================================================================
Milestone 3C.1 Task 2:
Scientifically profiles the active runtime path to isolate and quantify the latency
of every stage and access-control operation across all 5 operational cases:
  Case A: First encounter of a new stranger
  Case B: Repeated frames of an already tracked stranger
  Case C: Recognized enrolled person
  Case D: No-face frame
  Case E: Multi-face frame
"""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module1_face_recognition.anti_spoofing.deep_pad import MiniFASNetV2Detector
from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline

PROFILE_JSON = ROOT / "logs" / "benchmarks" / "m03c_1_face_runtime_profile.json"


def time_block(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return res, (t1 - t0) * 1000.0


def main():
    logger.info("=================================================================")
    logger.info("INTELLIGATE — MILESTONE 3C.1 FACE RUNTIME PATH PROFILING")
    logger.info("=================================================================")
    PROFILE_JSON.parent.mkdir(parents=True, exist_ok=True)

    db_path = ROOT / "profile_temp.db"
    if db_path.exists():
        db_path.unlink()

    store = EventStore(str(db_path))
    runtime = SoftwareRuntime(store)
    face_engine = FaceEngine(store)

    # Load 5 sample images from LFW
    lfw_dir = ROOT / "data" / "lfw" / "lfw"
    imgs = sorted(list(lfw_dir.rglob("*.jpg")))[:10]
    
    # Enrol person for Case C
    enrolled_img = cv2.imread(str(imgs[0]))
    face_engine.register("Enrolled_Alice", "employee", enrolled_img)
    logger.info("Enrolled 'Enrolled_Alice' in local gallery.")

    events_captured = []
    def callback(evt):
        events_captured.append(evt)
        runtime.observe(evt, trusted=True)

    pipeline = ReplayPipeline("face", runtime, callback, face_engine=face_engine, device="cpu")

    def make_canvas(raw_face, offset_x=0):
        canvas = np.full((480, 640, 3), 115, dtype=np.uint8)
        if raw_face is not None:
            h, w = raw_face.shape[:2]
            y = (480 - h) // 2
            x = (640 - w) // 2 + offset_x
            canvas[y : y + h, x : x + w] = raw_face
        return canvas

    enrolled_canvas = make_canvas(enrolled_img)
    stranger1_canvas = make_canvas(cv2.imread(str(imgs[1])))
    stranger2_canvas = make_canvas(cv2.imread(str(imgs[2])))
    no_face_canvas = make_canvas(None)
    
    # Multi-face canvas (two faces side-by-side)
    f1 = cv2.imread(str(imgs[3]))
    f2 = cv2.imread(str(imgs[4]))
    multi_canvas = np.full((480, 640, 3), 115, dtype=np.uint8)
    multi_canvas[100:350, 50:300] = cv2.resize(f1, (250, 250))
    multi_canvas[100:350, 340:590] = cv2.resize(f2, (250, 250))

    # Detailed sub-component breakdown for Case A (First encounter of stranger)
    subcomponents_case_a = {}
    
    raw_stranger_face = cv2.imread(str(imgs[5]))
    case_a_canvas = make_canvas(raw_stranger_face)

    # 1. Detection
    dets, t_det = time_block(face_engine.detect, case_a_canvas)
    subcomponents_case_a["1_yunet_detection_ms"] = round(t_det, 2)

    # 2. PAD
    f_bbox = [int(dets[0][0]), int(dets[0][1]), int(dets[0][2]), int(dets[0][3])]
    pad_res, t_pad = time_block(pipeline.pad_detector.evaluate_face, case_a_canvas, f_bbox)
    subcomponents_case_a["2_pad_minifasnetv2_ms"] = round(t_pad, 2)

    # 3. SFace Feature
    emb, t_feat = time_block(face_engine.feature, case_a_canvas, dets[0])
    subcomponents_case_a["3_sface_alignment_and_feature_ms"] = round(t_feat, 2)

    # 4. Gallery Match
    match_res, t_match = time_block(face_engine.match, emb)
    subcomponents_case_a["4_gallery_match_ms"] = round(t_match, 2)

    # 5. FaceEngine.stranger (includes AES-256 seal and SQLite cpu_strangers query/insert)
    t0_st = time.perf_counter()
    stranger_id = face_engine.stranger(emb)
    t_stranger = (time.perf_counter() - t0_st) * 1000.0
    subcomponents_case_a["5_stranger_db_and_aes_encryption_ms"] = round(t_stranger, 2)

    # 6. Event Dispatch & SoftwareRuntime Policy (includes event logging, review_requests, visitor_visits)
    test_evt = {
        "event_type": "stranger_alert",
        "module": "face",
        "identity": stranger_id,
        "confidence": 0.12,
        "method": "SFace",
        "camera_id": 0,
        "meta": {
            "pad_status": pad_res["pad_status"],
            "pad_score": pad_res["pad_score"],
            "pad_model": pad_res["pad_model"],
            "stranger_id": stranger_id,
            "session_id": "test_session",
            "provenance": "perception_internal",
            "trusted_perception": True,
        }
    }
    obs, t_runtime = time_block(runtime.observe, test_evt, trusted=True)
    subcomponents_case_a["6_software_runtime_policy_and_event_db_ms"] = round(t_runtime, 2)

    # 7. Overlay Drawing
    res_overlay, t_overlay = time_block(pipeline.overlay_frame, case_a_canvas)
    subcomponents_case_a["7_visualization_overlay_ms"] = round(t_overlay, 2)

    subcomponents_case_a["total_isolated_sum_ms"] = round(
        sum(subcomponents_case_a.values()), 2
    )

    logger.info("Subcomponent latency for single stranger encounter:")
    for k, v in subcomponents_case_a.items():
        logger.info(f"  {k}: {v} ms")

    # Now profile all 5 Operational Cases over 20 iterations
    case_results = {}
    
    # Case A: First encounter of fresh strangers (20 distinct strangers)
    lat_case_a = []
    for i in range(20):
        c_img = make_canvas(cv2.imread(str(imgs[i % len(imgs)])))
        pipeline.face_states.clear() # Force first encounter
        pipeline.tracker.tracks.clear()
        t0 = time.perf_counter()
        pipeline.process_frame(c_img)
        lat_case_a.append((time.perf_counter() - t0) * 1000.0)
    case_results["Case_A_First_Encounter_Stranger"] = {
        "mean_ms": round(float(np.mean(lat_case_a)), 2),
        "median_p50_ms": round(float(np.percentile(lat_case_a, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_case_a, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_case_a)), 2),
    }

    # Case B: Repeated frames of an already tracked stranger (20 consecutive frames)
    lat_case_b = []
    # Seed track
    pipeline.process_frame(stranger1_canvas)
    for _ in range(20):
        t0 = time.perf_counter()
        pipeline.process_frame(stranger1_canvas)
        lat_case_b.append((time.perf_counter() - t0) * 1000.0)
    case_results["Case_B_Repeated_Tracked_Stranger"] = {
        "mean_ms": round(float(np.mean(lat_case_b)), 2),
        "median_p50_ms": round(float(np.percentile(lat_case_b, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_case_b, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_case_b)), 2),
    }

    # Case C: Recognized enrolled person (20 frames, with tracking cache)
    lat_case_c = []
    pipeline.process_frame(enrolled_canvas)
    for _ in range(20):
        t0 = time.perf_counter()
        pipeline.process_frame(enrolled_canvas)
        lat_case_c.append((time.perf_counter() - t0) * 1000.0)
    case_results["Case_C_Recognized_Enrolled_Person"] = {
        "mean_ms": round(float(np.mean(lat_case_c)), 2),
        "median_p50_ms": round(float(np.percentile(lat_case_c, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_case_c, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_case_c)), 2),
    }

    # Case D: No-face frame (20 frames)
    lat_case_d = []
    for _ in range(20):
        t0 = time.perf_counter()
        pipeline.process_frame(no_face_canvas)
        lat_case_d.append((time.perf_counter() - t0) * 1000.0)
    case_results["Case_D_No_Face_Frame"] = {
        "mean_ms": round(float(np.mean(lat_case_d)), 2),
        "median_p50_ms": round(float(np.percentile(lat_case_d, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_case_d, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_case_d)), 2),
    }

    # Case E: Multi-face frame (2 faces, first encounter)
    lat_case_e = []
    for _ in range(20):
        pipeline.face_states.clear()
        pipeline.tracker.tracks.clear()
        t0 = time.perf_counter()
        pipeline.process_frame(multi_canvas)
        lat_case_e.append((time.perf_counter() - t0) * 1000.0)
    case_results["Case_E_Multi_Face_Frame_First_Encounter"] = {
        "mean_ms": round(float(np.mean(lat_case_e)), 2),
        "median_p50_ms": round(float(np.percentile(lat_case_e, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_case_e, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_case_e)), 2),
    }

    # Clean up temp db
    store.close()
    for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subcomponents_case_a": subcomponents_case_a,
        "cases": case_results,
    }

    PROFILE_JSON.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info(f"Saved profiling payload to {PROFILE_JSON}")

    print("\n" + "=" * 70)
    print("INTELLIGATE — RUNTIME PROFILING RESULTS ACROSS 5 OPERATIONAL CASES")
    print("=" * 70)
    for c_name, data in case_results.items():
        print(f"{c_name:<45} : Mean={data['mean_ms']:>6.2f} ms | p50={data['median_p50_ms']:>6.2f} ms | p95={data['p95_ms']:>6.2f} ms | {data['fps']:>6.2f} FPS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
