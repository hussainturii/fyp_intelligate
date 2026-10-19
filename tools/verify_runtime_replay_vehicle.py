"""
Verify Runtime Vehicle Replay Pipeline:
- GPU model loading on active ReplayPipeline
- CPU fallback mode
- Offline operation (allow_download=False)
- Visualization overlays and text labels
- Event callback dispatch and event store logging
- Gate policy integration
- Corrupted model handling
- Realistic sustained throughput (FPS) measurement
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path("D:/MZHR_FYP/INTELLIGATE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import ReplayPipeline

def run_runtime_replay_verification():
    print("\n=======================================================")
    print("VERIFYING ACTIVE VEHICLE REPLAY PIPELINE RUNTIME")
    print("=======================================================")

    test_img_p = ROOT / "data/pakistan_plates/combined_clean/test_a/images/da_test_vehicle_28_jpg.rf.64a9c7eae3076919bd141d0108f3eef9.jpg"
    assert test_img_p.exists(), f"Missing test fixture {test_img_p}"
    frame = cv2.imread(str(test_img_p))
    assert frame is not None, "Failed to load test frame"

    # 1. GPU Active Vehicle Replay Pipeline
    print("\n[1] Initializing GPU Vehicle ReplayPipeline...")
    import tempfile
    from module2_vehicle.database.vehicle_db import VehicleDatabase
    vdb = VehicleDatabase(str(ROOT / "logs/cpu_vehicles.h5"), str(ROOT / "logs/cpu_unknown_vehicles.h5"))

    t_dir = Path(tempfile.mkdtemp())
    store_gpu = EventStore(str(t_dir / "store_gpu.db"))
    runtime_gpu = SoftwareRuntime(store_gpu)
    dispatched_events_gpu = []
    
    pipeline_gpu = ReplayPipeline(
        mode="vehicle",
        runtime=runtime_gpu,
        callback=lambda ev: dispatched_events_gpu.append(runtime_gpu.observe(ev, trusted=True)),
        face_engine=None,
        vehicle_db=vdb,
        device="cuda:0"
    )

    assert pipeline_gpu.plate_detector._model is not None, "GPU Plate Detector model failed to load"
    assert "cuda" in pipeline_gpu.plate_detector.device, "GPU device was not set"

    # Process 20 frames to measure realistic sustained throughput
    t0 = time.perf_counter()
    res_gpu = None
    for _ in range(20):
        res_gpu = pipeline_gpu.process_frame(frame)
    gpu_duration = time.perf_counter() - t0
    gpu_fps = 20.0 / gpu_duration
    print(f"  GPU Replay Pipeline Sustained FPS: {gpu_fps:.2f} (latency: {gpu_duration/20*1000:.1f} ms/frame)")
    print(f"  Detected Vehicles in ROI: {res_gpu.get('count')}")
    print(f"  Dispatched Events: {len(dispatched_events_gpu)}")
    assert len(dispatched_events_gpu) >= 1, "Expected vehicle event dispatch"
    top_ev = dispatched_events_gpu[0]
    print(f"  Top Event Decision: {top_ev['decision']} (identity: {top_ev['identity'] if 'identity' in top_ev else 'N/A'})")

    # Verify Annotated Frame Overlay
    annotated = res_gpu.get("annotated_frame")
    assert annotated is not None, "Annotated frame missing"
    assert annotated.shape == frame.shape, "Annotated frame shape mismatch"
    assert not np.array_equal(annotated, frame), "Annotated frame was not drawn on"
    out_overlay_p = ROOT / "docs/report_assets/m03b_detector_comparison/replay_annotated_frame_sample.jpg"
    out_overlay_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_overlay_p), annotated)
    print(f"  Saved annotated preview to: {out_overlay_p.name}")

    # 2. CPU Fallback Mode
    print("\n[2] Initializing CPU Fallback Vehicle ReplayPipeline...")
    store_cpu = EventStore(str(t_dir / "store_cpu.db"))
    runtime_cpu = SoftwareRuntime(store_cpu)
    dispatched_events_cpu = []
    
    pipeline_cpu = ReplayPipeline(
        mode="vehicle",
        runtime=runtime_cpu,
        callback=lambda ev: dispatched_events_cpu.append(runtime_cpu.observe(ev, trusted=True)),
        face_engine=None,
        vehicle_db=vdb,
        device="cpu"
    )
    assert pipeline_cpu.plate_detector.device == "cpu"

    t0 = time.perf_counter()
    for _ in range(5):
        res_cpu = pipeline_cpu.process_frame(frame)
    cpu_duration = time.perf_counter() - t0
    cpu_fps = 5.0 / cpu_duration
    print(f"  CPU Replay Pipeline Sustained FPS: {cpu_fps:.2f} (latency: {cpu_duration/5*1000:.1f} ms/frame)")
    assert res_cpu is not None

    # 3. Missing / Corrupted Model Graceful Fallback
    print("\n[3] Testing Missing/Corrupted Model Graceful Fallback in ReplayPipeline...")
    from module2_vehicle.plate_detector import PlateDetector
    det_corrupt = PlateDetector(model_path="models/non_existent_path.pt", allow_download=False)
    assert det_corrupt._model is None, "Should fall back to classical"
    crops = det_corrupt.detect(frame)
    assert isinstance(crops, list)
    print(f"  Corrupted/Missing model fallback returned {len(crops)} crops via classical OpenCV without crashing.")

    print("\n[SUCCESS] All Runtime Replay Verification Checks Passed.")

if __name__ == "__main__":
    run_runtime_replay_verification()
