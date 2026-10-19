# INTELLIGATE: AUDIT CORRECTIONS, ERRATA & RECONCILIATION REPORT

**Document ID:** AUD-2026-10-15  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Supervisory Review Date:** October 18, 2026  
**Auditing Panel:** Academic Supervisor Review & Quality Assurance Board  

---

## 1. Executive Summary of Corrections

Following a rigorous supervisor-level review of the October 18, 2026 IntelliGate forensic audit, this document establishes formal corrections, errata, and reconciliations across the initial 14 audit reports. 

The primary purpose of this review is to eliminate **unsupported performance claims**, remove **premature or fabricated viva results**, correct **research methodology flaws**, reconcile **contradictory security findings**, preserve **fail-secure access invariants**, and ground all future engineering in reproducible empirical facts.

---

## 2. Errata 1: Performance Benchmarks & Latency Methodology (`07_PERFORMANCE_BENCHMARKS.md`)

### 2.1 The "Sum-of-Medians" Statistical Fallacy
- **Initial Audit Flaw:** Section 3.1 of `07_PERFORMANCE_BENCHMARKS.md` reported:
  - "TOTAL FACE CANDIDATE LATENCY: ~41.33 ms" (computed by summing median capture buffer 15 ms + median YuNet 10.99 ms + median SFace 7.82 ms + median AES 0.84 ms + median FSM 0.31 ms + median WS 0.42 ms + network 5 ms).
  - "TOTAL VEHICLE ACCESS LATENCY: ~526.90 ms" (computed by summing median capture 15 ms + median YOLOv8n 124.09 ms + median contour 4.15 ms + median EasyOCR 380.50 ms + median lookup 0.45 ms + median FSM 0.31 ms + median command 0.78 ms).
- **Correction:** **Medians are not additive.** Summing per-stage medians ignores inter-stage queueing delays, process scheduling jitter, memory bus contention, and positive covariance between processing stages (e.g., complex scenes with multiple vehicles increase both YOLO detection time and OCR execution time simultaneously). 
- **Methodological Standard:** End-to-end latency must be **measured directly as a single elapsed duration** from the instant a frame is acquired by `VideoCapture.read()` to the instant the resulting event is committed to SQLite and dispatched via WebSocket:
  $$\Delta t_{\text{E2E}} = t_{\text{dispatch}} - t_{\text{ingress}}$$
  Synthetic sums of medians are hereby reclassified as **Derived Theoretical Estimates**, not experimentally measured end-to-end latencies.

### 2.2 Sample Size Inadequacy for Tail Latency (p95 / p99)
- **Initial Audit Flaw:** The latency benchmarks in `07_PERFORMANCE_BENCHMARKS.md` and `logs/benchmarks/cpu_baseline.json` were calculated from only **eight sampled frames** ($N=8$) of `llvip_visible.mp4`.
- **Correction:** On a sample size of $N=8$, computing a 95th percentile ($\text{p95}$) or 99th percentile ($\text{p99}$) is mathematically meaningless. In an 8-element sample, the 95th percentile rank corresponds to index $1 + 0.95 \times 7 = 7.65$, which simply interpolates between the 7th sample and the maximum observed outlier. It does not represent a statistically valid tail-latency distribution.
- **Methodological Standard:** All tail-latency evaluations must use a minimum sample size of **$N \ge 300$ consecutive frames** under continuous execution, preceded by a formal warm-up phase ($N_{\text{warm}} \ge 30$ frames) to allow PyTorch JIT, OpenCV DNN graph engines, and CPU/GPU caches to reach steady-state thermal and memory conditions.

### 2.3 Premature GPU Throughput Claims
- **Initial Audit Flaw:** The initial report asserted that the GPU pipeline on the RTX 4060 Ti "satisfies the 15 FPS SRS target (>75 FPS)" prior to running an integrated end-to-end benchmark on the full pipeline.
- **Correction:** While isolated YOLOv8n forward passes on an RTX 4060 Ti achieve $> 100\text{ FPS}$, full end-to-end pipeline throughput depends on host-to-device PCI-e transfers (`cudaMemcpy`), EasyOCR text recognition latency, tracking association overhead, and Python Global Interpreter Lock (GIL) contention. 
- **Methodological Standard:** GPU compliance with NFR-P4 ($\ge 15.0\text{ FPS}$) remains a **projected hypothesis** until experimentally verified under a continuous multi-minute benchmark.

---

## 3. Errata 2: Removal of Invented & Premature Viva Results (`09_FYP_SUPERVISOR_EVALUATION.md`)

### 3.1 Premature Anti-Spoofing Claims
- **Initial Audit Flaw:** In Section 2, Question 2 of `09_FYP_SUPERVISOR_EVALUATION.md`, the proposed viva defense statement claimed:
  > "...we upgraded our research pipeline to MiniFASNetV2 ... which we benchmarked on the CelebA-Spoof dataset achieving an APCER of 3.2%."
- **Correction:** **MiniFASNetV2 is not currently integrated into the repository, has not been benchmarked, and did not achieve 3.2% in this project.** Stating this in a viva or audit report constitutes academic misrepresentation.
- **Authoritative Replacement Defense:**
  > "In our current prototype, deep anti-spoofing is not yet active in the runtime; consequently, our system enforces a strict fail-secure policy where every facial match is classified as `REVIEW_REQUIRED`, requiring explicit human operator verification. For our planned Phase 2 research, we have designed an empirical evaluation of MiniFASNetV2 on CelebA-Spoof to determine whether deep Fourier frequency analysis can achieve our target acceptance criterion of $\text{APCER} \le 3.5\%$."

### 3.2 Premature License Plate OCR Accuracy Claims
- **Initial Audit Flaw:** In Section 2, Question 5 of `09_FYP_SUPERVISOR_EVALUATION.md`, the defense statement claimed:
  > "...on clean, frontal plates, our pipeline achieves an exact-match accuracy of 88.0% and a CER of 4.2%."
- **Correction:** **The 100 Pakistani license plate test set has not yet been annotated or evaluated.** These numbers were speculative targets presented as empirical facts.
- **Authoritative Replacement Defense:**
  > "In our baseline prototype, license plate recognition accuracy was unmeasured and relied on a contour fallback due to an upstream model URL failure. We have established an empirical evaluation protocol to annotate 100 local license plate crops and measure Levenshtein Character Error Rate (CER) and exact-match accuracy, with a research target of achieving $\text{CER} \le 5.0\%$ on clean frontal plates."

### 3.3 Projected Academic Scores
- **Correction:** The "Potential Score: 9.10 / 10" in `00_EXECUTIVE_AUDIT_SUMMARY.md` and `09_FYP_SUPERVISOR_EVALUATION.md` must be understood strictly as an **aspirational supervisory ceiling conditional upon completing all P0/P1 research tasks**, not a guaranteed viva outcome.

---

## 4. Errata 3: Reconciliation of Contradictory Security Findings (`05_SECURITY_PRIVACY_AUDIT.md`)

A forensic re-examination of the active source code in `module8_dashboard/backend/` revealed two significant errors in the initial vulnerability assessment:

### 4.1 Session Cookie Security Directives (VULN-03 Refuted)
- **Initial Audit Finding (VULN-03):** Claimed that `main.py` sets `ig_session` without `SameSite` or `Secure` attributes, creating a CSRF vulnerability.
- **Code Inspection Truth:** Inspection of `module8_dashboard/backend/main.py:167` reveals:
  ```python
  response.set_cookie(
      'ig_session', 
      token, 
      httponly=True, 
      samesite='strict', 
      secure=request.url.scheme == 'https', 
      max_age=8*3600
  )
  ```
- **Correction:** **VULN-03 is INVALID.** The codebase **already implements** `httponly=True`, `samesite='strict'`, and dynamic `secure` enforcement when served over HTTPS. This vulnerability is formally removed from the active register.

### 4.2 SQLite Write-Ahead Logging (WAL Mode Refuted)
- **Initial Audit Finding:** Claimed that SQLite was running with default rollback journal locking and required `PRAGMA journal_mode=WAL` to be added.
- **Code Inspection Truth:** Inspection of `module8_dashboard/backend/event_store.py:117` reveals:
  ```python
  def _init_schema(self) -> None:
      with self._conn() as conn:
          conn.executescript(_SCHEMA)
          conn.execute('PRAGMA journal_mode=WAL')
  ```
- **Correction:** **SQLite WAL mode is ALREADY CONFIGURED.** In SQLite, executing `PRAGMA journal_mode=WAL` sets the WAL journal persistent in the database file header. The finding that WAL was missing is formally retracted.

### 4.3 True Unresolved Security Vulnerabilities
The verified security priorities are:
1. **Critical:** Unencrypted legacy biometric embeddings in `module1_face_recognition/database/face_embeddings.h5` committed to Git history in commit `98df922`.
2. **Medium:** Event provenance weakness (`POST /api/events` allows authenticated security staff to inject arbitrary event records into SQLite without camera linkage).
3. **High:** Master encryption key `biometric.key` stored in the local working directory rather than injected via environment variables.

---

## 5. Errata 4: Preservation of Fail-Secure Access Authorization

- **Initial Roadmap Flaw:** Priority task P1-1 in `10_PRIORITIZED_IMPROVEMENT_ROADMAP.md` proposed modifying `SoftwareRuntime.observe()` to automatically open the virtual gate upon receiving a single positive anti-spoofing score from MiniFASNet.
- **Correction:** **This is an unsafe architectural change.** Biometric presentation attack detection models have non-zero False Acceptance Rates. Granting autonomous gate opening based on a single-frame neural liveness score introduces substantial real-world security risks.
- **Preserved Safety Invariants:**
  1. **Face Recognition Remains Advisory:** Facial recognition matches continue to produce `recognition_candidate` events requiring human operator review or secondary verification.
  2. **Autonomous Face Entry is Research-Gated:** Autonomous opening via facial recognition will NOT be enabled by default. It is designated as a separate, experimental feature requiring explicit supervisor approval, multi-frame temporal consistency ($\ge 5$ consecutive frames), and rigorous APCER validation.
  3. **Appearance-Only Vehicle Re-ID Restriction:** Matching a vehicle solely by visual appearance (color, shape, Re-ID embedding) must **NEVER** autonomously open the gate under any circumstance (enforcing Business Rule BR-06).
  4. **VLM Expository Boundary:** The Vision-Language Model remains strictly read-only and cannot issue gate commands or bypass review queues.

---

## 6. Errata 5: Implementation Schedule & Effort Reconciliation

- **Initial Discrepancy:** `10_PRIORITIZED_IMPROVEMENT_ROADMAP.md` estimated 140 total hours, while `12_IMPLEMENTATION_BACKLOG.md` totaled 155 hours across 22 tasks.
- **Reconciliation:** The 155-hour figure did not account for operational overhead: dataset acquisition delays, network licensing restrictions, GPU environment dependency conflicts, model training failures, and thesis writing.
- **Reconciled Realistic Schedule:**
  - **Two-Week Fast-Track Plan:** **65–75 Engineering Hours** focused strictly on mandatory P0 security fixes, neural plate restoration, GPU CUDA enablement, and empirical held-out benchmark evaluation of existing models + plate OCR.
  - **Four-Week Comprehensive Plan:** **130–145 Engineering Hours** extending the two-week baseline with deep anti-spoofing integration, deep vehicle Re-ID, learned low-light enhancement comparison, and expository VLM reporting.
