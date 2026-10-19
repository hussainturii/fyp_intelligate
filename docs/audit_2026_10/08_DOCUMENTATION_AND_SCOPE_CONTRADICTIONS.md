# INTELLIGATE: DOCUMENTATION CLAIMS AUDIT & SCOPE CONTRADICTION REGISTER

**Document ID:** AUD-2026-10-08  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Documents Audited:** Original FYP Proposal (Nov 2025), Detailed Proposal v1.0 (Dec 2025), IntelliGate SRS v1.0 & v3.0 (Sep 2026), SRS Evaluation Slides, IntelliGate 75% Evaluation Report (Oct 2026), IntelliGate Final Report Draft (Oct 2026), Technical Documentation v3.0, README files.  
**Audit Date:** October 18, 2026  

---

## 1. Documentation Audit Methodology

Statements made across project documents are evaluated against the empirical ground truth of the active repository checkout (`c909145` / `ad76140`). Claims are classified into six rigorous forensic categories:
1. **Accurate:** Claim is fully substantiated by executable code, tests, and data in the repository.
2. **Partially Accurate:** Claim reflects true aspects of the system, but omits critical constraints, degraded modes, or missing evaluation.
3. **Unsupported:** Claim asserts quantitative metrics (e.g. accuracy percentages) or capabilities for which zero empirical evidence exists in the repository.
4. **Misleading:** Claim uses terminology that implies advanced AI techniques (e.g. "multimodal VLM" or "thermal sensing") when only classical heuristics or templates exist.
5. **Contradictory:** Document makes claims that directly conflict with other statements in the same document or between historical drafts.
6. **Outdated:** Claim was valid in an earlier design phase but was rendered obsolete by subsequent scope amendments (e.g. physical hardware wiring).

---

## 2. Comprehensive Documentation Claims Audit

| Claim ID | Source Document & Section | Documented Claim | Repository Ground Truth | Forensic Classification | Risk Level | Academic Defensibility Impact |
|---|---|---|---|---|---|---|
| **CLM-01** | SRS v1.0 §10.5; Tech Doc v3 §3.3 | "System interfaces with Raspberry Pi 4B GPIO pins driving 5V optocoupled relays to actuate the physical gate." | The system is a pure software prototype. No Raspberry Pi, GPIO libraries, or relay circuits exist in the repository. | **OUTDATED / CONTRADICTORY** | **HIGH** | If examiners evaluate against this hardware claim, marks will be deducted for non-delivery of physical hardware. |
| **CLM-02** | SRS v1.0 §10.6; Proposal §1.3 | "Mandatory Long-Wave Infrared (LWIR) dual-spectrum camera rig with automatic optical switching." | No physical LWIR sensor exists. The system ingests standard webcams and recorded LLVIP infrared replay video clips. | **OUTDATED / CONTRADICTORY** | **HIGH** | Must be formally superseded by software-only recorded dataset replay amendment. |
| **CLM-03** | SRS v1.0 §4.1; Proposal FE-1 | "Face recognition is powered by a fine-tuned Vision Transformer (ViT-Base/16) extracting 768-D embeddings." | The active runtime uses OpenCV Zoo SFace ONNX (128-D). The legacy `face_recognizer.py` has an off-the-shelf ImageNet ViT that was never fine-tuned on faces. | **CONTRADICTORY / MISLEADING** | **CRITICAL** | Claiming a fine-tuned ViT in viva when SFace ONNX is running is an immediate viva failure point. |
| **CLM-04** | SRS v1.0 §4.2; Proposal FE-4 | "Deep Vehicle Re-Identification using OSNet metric learning trained on the VeRi-776 benchmark." | The active runtime extracts 768-D concatenated RGB color histograms (`cv2.calcHist`). Zero OSNet models exist in active code. | **MISLEADING / UNSUPPORTED** | **CRITICAL** | Claiming deep metric vehicle Re-ID when a color histogram is executing will be severely penalized. |
| **CLM-05** | SRS v1.0 §4.3; Proposal FE-2 | "Low-light analytics algorithm recovers thermal heat signatures from optical RGB video feeds." | Optical CMOS sensors cannot detect long-wave thermal radiation. The system uses OpenCV CLAHE contrast stretching. | **MISLEADING / UNSCIENTIFIC** | **CRITICAL** | Violates basic optical physics. Examiners will challenge students on electromagnetic spectrum fundamentals. |
| **CLM-06** | SRS v1.0 §4.4; Proposal FE-3 | "Pedestrian intent and behavioral anomalies are predicted using State Space Models (Mamba / S4)." | Zero State Space Models exist. Loitering is flagged using a basic rule: continuous track duration $> 30$ seconds. | **MISLEADING / UNSUPPORTED** | **HIGH** | Substituted heuristic must be disclosed; SSM claim must be removed or moved to Future Work. |
| **CLM-07** | SRS v1.0 §4.7; Proposal FE-1 | "Automated visual incident reporting powered by a local Vision-Language Model (LLaVA-1.5 7B)." | Reports are deterministic string templates formatting SQL count queries (`'method': 'event-template (not a VLM)'`). | **MISLEADING / UNSUPPORTED** | **HIGH** | The prototype code is honest, but older project documents still promise a generative VLM. |
| **CLM-08** | SRS v1.0 §4.2; NFR-P2 | "License plate recognition achieves $\ge 99.4\%$ accuracy." | OCR accuracy has never been measured on an annotated test split. Neural plate detection failed with 404 URL, forcing contour fallback. | **UNSUPPORTED / FABRICATED TARGET** | **CRITICAL** | High-accuracy claims without an empirical evaluation dataset are treated as fabricated in academic vivas. |
| **CLM-09** | SRS v1.0 §4.2; NFR-P3 | "Vehicle make/model Re-Identification accuracy $\ge 97.0\%$." | Re-ID was never evaluated on held-out test splits. Furthermore, Re-ID is measured via Rank-1/mAP, not classification accuracy. | **UNSUPPORTED / METRIC CONFUSION** | **HIGH** | Confuses vehicle classification (make/model) with metric identity retrieval (Re-ID). |
| **CLM-10** | SRS v1.0 §4.10; Proposal DB-1 | "Database architecture built on PostgreSQL with TimescaleDB time-series and pgvector vector search." | The system runs on a single local SQLite 3 database (`logs/events.db`) using basic SQL tables and Python loops for vector matching. | **CONTRADICTORY / OUTDATED** | **MEDIUM** | SQLite was chosen for standalone portability. Must be documented as an approved architectural deviation. |
| **CLM-11** | SRS v1.0 §5.4; BR-05 | "Physical emergency stop halts gate motor within 200 ms with certified physical braking distance." | The virtual gate state machine halts internal software state within 1 ms. No physical braking distance exists or can be measured. | **OUTDATED / SCOPE MISMATCH** | **MEDIUM** | Must be reframed as software state-machine interrupt latency, not certified physical actuator safety. |
| **CLM-12** | Tech Doc v3 §24; SRS §5.4 | "Biometric facial embeddings stored in the system cannot be reverse-engineered or inverted." | Facial embeddings CAN be partially reconstructed via deep generative inversion attacks (e.g. Vec2Face). | **MISLEADING / INACCURATE** | **MEDIUM** | Biometric vectors must be defended as encrypted data at rest, not mathematically irreversible hashes. |
| **CLM-13** | Proposal & SRS UI Mockups | "Dashboard mockups show 99.4% OCR accuracy, 97.8% Re-ID confidence, and 0.3s gate decisions." | These values were illustrative UI graphic design wireframes, but were misinterpreted in later drafts as measured results. | **MISLEADING** | **HIGH** | Must be explicitly labeled as UI wireframes, not operational system metrics. |
| **CLM-14** | Tech Doc v3 References | "YOLOv4 citation credited to 'Jochko et al. (2020)'. LLaVA citation dated 2021." | YOLOv4 is by Alexey Bochkovskiy et al. (2020). LLaVA was published at NeurIPS in December 2023. | **OUTDATED / CITATION ERRORS** | **LOW** | Academic bibliography errors undermine research credibility during evaluation. |

---

## 3. Approved Scope Reconciliation & Replacement Wording

To prevent academic penalties and eliminate discrepancies before the final evaluation, the following **authoritative replacement statements** must be adopted in the Final Report and defense slides:

### 3.1 Replacement for Hardware & Sensor Claims (CLM-01, CLM-02, CLM-11)
- **Previous Unsupported Text:**
  > "IntelliGate is deployed on an edge Raspberry Pi 4B connected to physical 5V relays and dual-spectrum RGB/LWIR optical sensors. An emergency stop halts the physical gate motor within 200 ms with verified stopping distances."
- **Authoritative Approved Replacement:**
  > "IntelliGate is developed as a high-integrity, software-only decision support prototype. Physical gate mechanics are emulated using a deterministic virtual finite state machine running safety interlocks, emergency stop pre-emption, and restart state recovery. Optical sensing is evaluated using standard USB/integrated webcams, uploaded surveillance video clips, and curated multi-modal benchmark datasets (such as LLVIP for paired visible and infrared analysis). The system establishes a modular architecture capable of interfacing with industrial gate controllers via standard network adapters (HTTP/MQTT), while remaining fully independent of custom physical hardware during this evaluation phase."

### 3.2 Replacement for Face & Vehicle Model Claims (CLM-03, CLM-04)
- **Previous Unsupported Text:**
  > "Face recognition is powered by a fine-tuned Vision Transformer (ViT-Base/16) extracting 768-D embeddings. Obscured vehicles are identified via deep metric Re-Identification achieving 97% accuracy on VeRi-776."
- **Authoritative Approved Replacement:**
  > "Face detection and recognition are implemented using task-pretrained deep convolutional neural networks (OpenCV Zoo YuNet for landmark-guided face detection and SFace for 128-dimensional metric feature extraction). Biometric templates are cryptographically encrypted at rest with AES-256-GCM. In the initial software baseline, vehicle visual appearance is modeled using 768-D concatenated color histograms. Deep metric learning Re-Identification (OSNet) and fine-tuned Vision Transformers represent planned research enhancements evaluated under GPU acceleration in Phase 2."

### 3.3 Replacement for Night Vision & Thermal Claims (CLM-05)
- **Previous Unsupported Text:**
  > "The night vision analytics module utilizes Contrast Limited Adaptive Histogram Equalization to reconstruct thermal heat signatures from low-light optical video feeds."
- **Authoritative Approved Replacement:**
  > "The night vision module evaluates illumination changes via real-time luminance profiling. Under low-light conditions (luminance $< 35$), spatial contrast is enhanced using Contrast Limited Adaptive Histogram Equalization (CLAHE). The system explicitly distinguishes optical RGB contrast enhancement from physical thermal imaging: optical sensors cannot recover long-wave infrared radiation. Thermal pedestrian analytics are evaluated independently by ingesting genuine long-wave infrared video replays from the LLVIP dataset, with explicit warnings regarding domain shift when processed by visible-trained object detectors."

### 3.4 Replacement for Behavioral & VLM Claims (CLM-06, CLM-07)
- **Previous Unsupported Text:**
  > "Pedestrian behavioral anomalies and loitering are predicted using State Space Models (Mamba). Daily security incident summaries are automatically authored by a local LLaVA-1.5 multimodal Vision-Language Model."
- **Authoritative Approved Replacement:**
  > "Behavioral analysis and stranger escalation are enforced through deterministic operational rules: continuous track dwell exceeding 30 seconds triggers loitering alerts, while unknown visitors appearing three distinct times within 24 hours escalate to high-severity repeat visitor alerts. Daily incident summaries are generated using structured event aggregation templates directly linked to cryptographic event IDs, eliminating generative model hallucinations while preserving verifiable audit trails. Integration of local Vision-Language Models (such as Moondream2) represents an expository reporting extension evaluated on dedicated GPU hardware."

### 3.5 Replacement for Accuracy Target Claims (CLM-08, CLM-09)
- **Previous Unsupported Text:**
  > "The system achieves 99.4% license plate recognition accuracy, 97% vehicle Re-ID accuracy, and 99.5% annual system uptime."
- **Authoritative Approved Replacement:**
  > "The system targets high-precision automated access control with human-in-the-loop review boundaries. License plate recognition combines YOLO-based localization with EasyOCR character recognition. Under unconstrained outdoor surveillance, uncalibrated high-accuracy claims ($>99\%$) are avoided; instead, empirical character error rates (CER) and exact-match rates are measured on designated test sets. System reliability is validated through continuous automated regression testing and soak testing rather than theoretical annual availability claims."
