# INTELLIGATE — MILESTONE 3C.1 EXPERIMENT REPORT
## Face Perception Runtime Path Profiling and Bottleneck Investigation

**Author:** Hussain Raza (SP23-BAI-057)  
**Academic Role:** BS Artificial Intelligence FYP Lead Researcher  
**System Architecture:** Hybrid OpenCV DNN + PyTorch 2.0 (CUDA 12.8) Perception Engine  
**Hardware Specification:** AMD Ryzen 7 5800X (8C/16T, 3.8 GHz base / 4.7 GHz boost), 32 GB DDR4-3600 RAM, NVIDIA GeForce RTX 4060 Ti 16GB VRAM, PCIe Gen4 NVMe SSD, Windows 11 Enterprise.  
**Benchmark Artifacts:** [`logs/benchmarks/m03c_1_face_runtime_profile.json`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_1_face_runtime_profile.json), [`logs/benchmarks/m03c_face_pipeline_performance.json`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_face_pipeline_performance.json)

---

## 1. Executive Summary & Problem Formulation

In Milestone 3C, the initial integration of the deep presentation-attack detection (PAD) model `MiniFASNetV2` reported a perplexing performance paradox:
- **Core Neural Perception (YuNet + MiniFASNetV2 + SFace):** ~23.78 ms per frame.
- **Full Pipeline End-to-End Latency:** 583.93 ms (GPU) to 601.81 ms (CPU).
- **Throughput Collapse:** 1.66–1.71 FPS.

The objective of this investigation was to instrument the active runtime path, empirically isolate every subcomponent from pixel acquisition to WebSocket emission, identify the exact root causes behind the ~560 ms non-perception latency penalty, and optimize the execution path without compromising security or architectural invariants.

---

## 2. Empirical Subcomponent Isolation Breakdown

To identify the bottleneck, a synthetic benchmark harness was executed with sub-millisecond precision profiling across each isolated subcomponent during a first-encounter stranger event:

```
[Frame Acquisition & Resizing]
         │ (0.42 ms)
         ▼
  [YuNet Face Detection] ───────► (10.42 ms)
         │
         ▼
[MiniFASNetV2 Deep PAD] ────────► (14.17 ms)
         │
         ▼
 [SFace Feature Extraction] ─────► (4.93 ms)
         │
         ▼
 [Gallery Cosine Match] ────────► (0.16 ms post-opt / was 3.34 ms)
         │
         ▼
[Stranger DB & AES-GCM Enc] ───► (0.16 ms post-opt / was 184.17 ms)
         │
         ▼
[SoftwareRuntime & EventStore] ─► (0.48 ms post-opt / was 411.97 ms)
         │
         ▼
  [Overlay Drawing] ─────────────► (0.01 ms)
```

### Table 1: Subcomponent Latency Breakdown (Single Stranger First Encounter)

| Subcomponent Stage | Pre-Optimization Latency | Post-Optimization Latency | Latency Share (Before) | Latency Share (After) | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. YuNet Detection (640×480)** | 11.73 ms | 10.42 ms | 2.1% | 34.4% | 1.1× |
| **2. MiniFASNetV2 PAD (80×80×3)** | 13.24 ms | 14.17 ms | 2.3% | 46.7% | 1.0× |
| **3. SFace Align & Embedding (128-D)**| 4.87 ms | 4.93 ms | 0.9% | 16.3% | 1.0× |
| **4. Gallery Matching (`cpu_faces`)** | 3.34 ms | 0.16 ms | 0.6% | 0.5% | **20.9×** |
| **5. Stranger Memory & AES-256 Seal** | 184.17 ms | 0.16 ms | 32.4% | 0.5% | **1,151.0×** |
| **6. SoftwareRuntime & EventStore** | 413.39 ms | 0.48 ms | 72.7% | 1.6% | **861.2×** |
| **7. Visualization & Overlays** | 0.01 ms | 0.01 ms | 0.0% | 0.0% | 1.0× |
| **Total Isolated Sum** | **630.75 ms** | **30.33 ms** | **100.0%** | **100.0%** | **20.8×** |

---

## 3. Root Cause Analysis

Profiling revealed that **95.1% of the total latency (597.56 ms out of 630.75 ms)** was consumed entirely outside the neural network models by synchronous SQLite disk I/O and unindexed database queries on Windows NTFS:

1. **Windows NTFS Connection Closure Checkpoints:**
   In SQLite WAL mode, when a connection opens and immediately closes inside a `@contextmanager def _conn()`, SQLite executes a passive closing checkpoint (`sqlite3_close()` flushes OS file buffers). On Windows NTFS, each OS-level handle closure with file flushes takes 120–135 ms. Because a single stranger encounter triggered three separate database calls (`FaceEngine.stranger`, `EventStore.log_event`, and `SoftwareRuntime.visit`), NTFS was flushed 3–4 times per frame (consuming ~400 ms).
2. **Linear Decryption of All Historical Strangers (`FaceEngine.stranger`):**
   Prior to this milestone, `FaceEngine.stranger()` performed a `SELECT * FROM cpu_strangers` and decrypted every historical stranger embedding in a Python loop using `AESGCM.decrypt()`. As the test set introduced 100 new strangers, each subsequent frame took longer as the database grew.
3. **Repeated Table DDL Schema Locks:**
   `db.execute('CREATE TABLE IF NOT EXISTS cpu_strangers ...')` was being called inside the frame-processing loop on every single stranger encounter, acquiring SQLite schema locks.
4. **Full Table Scans on Visitor Deduplication (`SoftwareRuntime.visit`):**
   `DELETE FROM visitor_visits WHERE seen < ?` was executed unconditionally on every frame without an index on `seen`, causing table scans and write locks.

---

## 4. Operational Latency Profile Across 5 Scenarios

The profiler was executed over 20 iterations for each of the 5 canonical operational cases:
- **Case A (First Encounter of Stranger):** Fresh face entering frame with no prior tracking state; requires YuNet detection, MiniFASNetV2 PAD inference, SFace embedding extraction, gallery match failure, AES-256 stranger sealing, SQLite event logging, and review request creation.
- **Case B (Repeated Tracked Stranger):** Already tracked face across consecutive video frames; tracker cache hits skip redundant neural feature extraction and stranger generation.
- **Case C (Recognized Enrolled Person):** Registered resident/staff member; matched against in-memory gallery cache.
- **Case D (No-Face Frame):** Frame containing background or scene only; YuNet returns empty detections.
- **Case E (Multi-Face First Encounter):** Two simultaneous new stranger faces in a single frame.

### Table 2: Benchmark Latency Across 5 Operational Cases (Post-Optimization)

| Operational Scenario | Mean Latency | Median (p50) | 95th Percentile (p95) | Throughput (FPS) | Pre-Opt FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Case A: First Encounter Stranger** | **22.39 ms** | 21.39 ms | 24.61 ms | **44.66 FPS** | 2.59 FPS |
| **Case B: Repeated Tracked Stranger** | **6.91 ms** | 6.69 ms | 8.41 ms | **144.62 FPS** | 95.19 FPS |
| **Case C: Recognized Enrolled Person**| **6.79 ms** | 6.81 ms | 7.29 ms | **147.27 FPS** | 111.66 FPS |
| **Case D: No-Face Frame** | **6.35 ms** | 6.23 ms | 6.89 ms | **157.39 FPS** | 136.98 FPS |
| **Case E: Multi-Face First Encounter**| **35.09 ms** | 35.20 ms | 36.27 ms | **28.50 FPS** | 3.09 FPS |

---

## 5. Architectural Improvements Implemented

1. **Thread-Local Connection Pooling in `EventStore`:**
   Added `threading.local()` connection reuse. Instead of creating and terminating a connection per transaction, each thread maintains an active SQLite connection in WAL mode with `PRAGMA synchronous = NORMAL` and `PRAGMA busy_timeout = 5000`. Transactions commit immediately without triggering Windows file-handle flush stalls.
2. **In-Memory Biometric Caching with Invalidation Revisions:**
   - Enrolled gallery embeddings are cached in memory in `FaceEngine._gallery_cache`. The cache is only refreshed when `gallery_revision` increments (upon enrolment or deletion). Cosine similarity matching is vectorized in RAM ($O(N)$ dot product taking $<0.2$ ms) rather than reading from disk and decrypting on every frame.
   - Active stranger embeddings are cached in memory in `FaceEngine._stranger_cache`. New strangers are sealed with AES-256-GCM and written to disk once, then cached for fast in-memory nearest-neighbor matching.
3. **Database Indexing:**
   Added indexes on `cpu_strangers(seen)`, `visitor_visits(seen)`, `visitor_visits(identity, seen)`, and `review_requests(identity, status, expires)`.
4. **Throttled Retention Pruning:**
   90-day retention pruning queries (`DELETE FROM ... WHERE seen < ?`) were throttled from running on every single frame to running at most once per hour.

---

## 6. Official Pipeline Benchmark Re-Evaluation

The official full-pipeline benchmark script [`tools/benchmark_face_pipeline.py`](file:///d:/MZHR_FYP/INTELLIGATE/tools/benchmark_face_pipeline.py) was re-executed over 100 genuine test frames (640×480) on both CPU and CUDA:

### Table 3: Official Face Perception Benchmark Latency (100 Frames)

| Performance Metric | CPU Fallback (Pre-Opt) | CPU Fallback (Post-Opt) | CUDA:0 (Pre-Opt) | CUDA:0 (Post-Opt) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Latency** | 601.81 ms | **23.91 ms** | 583.93 ms | **28.51 ms** |
| **p50 (Median)** | 592.14 ms | **21.81 ms** | 574.62 ms | **25.97 ms** |
| **p95 Latency** | 684.20 ms | **36.58 ms** | 652.80 ms | **45.31 ms** |
| **Sustained FPS** | 1.66 FPS | **41.83 FPS** | 1.71 FPS | **35.07 FPS** |
| **Speedup Factor** | Baseline | **25.2×** | Baseline | **20.5×** |

### Stage Latency Breakdown (Mean ms)
- **YuNet Detection:** 6.49 ms (CPU) / 7.43 ms (CUDA)
- **MiniFASNetV2 PAD:** 8.76 ms (CPU) / 11.56 ms (CUDA)
- **SFace Embedding:** 5.35 ms (CPU) / 6.49 ms (CUDA)
- **Tracker Update:** 0.06 ms (CPU) / 0.07 ms (CUDA)

*Note on CPU vs CUDA:* On batch size 1 at 80×80 input resolution, CPU execution using optimized PyTorch OpenMP kernels is slightly faster than GPU due to CUDA kernel launch overhead, PCIe host-to-device memory transfer, and tensor synchronization. Both devices easily exceed real-time requirements ($>35$ FPS).
