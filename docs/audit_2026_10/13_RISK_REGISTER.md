# INTELLIGATE: COMPREHENSIVE PROJECT RISK REGISTER & MITIGATION MATRIX

**Document ID:** AUD-2026-10-13  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Audit Date:** October 18, 2026  
**Auditing Standards:** ISO 31000:2018 Risk Management Guidelines & Project Management Body of Knowledge (PMBOK)  

---

## 1. Risk Scoring Methodology

Risks are quantified on a standard $5 \times 5$ matrix:
- **Likelihood ($L$):** 1 (Rare), 2 (Unlikely), 3 (Possible), 4 (Likely), 5 (Almost Certain)
- **Impact ($I$):** 1 (Insignificant), 2 (Minor), 3 (Moderate), 4 (Major), 5 (Catastrophic)
- **Risk Score ($R = L \times I$):**
  - **Critical (20–25):** Immediate project failure or academic rejection; must be mitigated immediately.
  - **High (12–19):** Significant grade penalty, security compromise, or major demo disruption.
  - **Medium (6–11):** Technical debt or minor scope reductions; manageable with standard controls.
  - **Low (1–5):** Cosmetic or documentation polish items.

---

## 2. Master Project Risk Register

| Risk ID | Category | Risk Description | Likelihood ($L$) | Impact ($I$) | Score ($R$) | Risk Level | Detection Indicator | Mitigation Strategy | Contingency Plan | Owner |
|---|---|---|---|---|---|---|---|---|---|---|
| **RSK-01** | Academic / Viva | **Failure to Justify SRS Model Divergences (ViT/SSM/LLaVA):** Viva panel penalizes candidates for claiming complex foundation models in reports while running ONNX CNNs and heuristics in the code. | 5 | 5 | **25** | **CRITICAL** | Examiners ask for ViT attention map visualizations or SSM equation derivations during defense. | Submit formal Scope Revision Note signed by supervisors; adopt authoritative replacement wording in thesis. | Transparently defend SFace and CNNs on edge latency and CASIA-WebFace benchmark metrics; present empirical ablation. | Lead Author / Supervisors |
| **RSK-02** | Security / Legal | **Biometric Data Exposure in Public Git History:** Committed `face_embeddings.h5` exposed in initial commit `98df922` violates GDPR/privacy laws. | 4 | 5 | **20** | **CRITICAL** | Security audit tools (`trufflehog`, `git-secrets`) flag committed biometric files. | Execute `git-filter-repo` to permanently erase `.h5` files from all Git commits; enforce `.gitignore` rules. | Keep repository strictly private until commit history is rewritten and verified clean. | Security Engineer |
| **RSK-03** | Academic / AI | **Evaluation Void (Lack of Held-Out Benchmark Metrics):** Thesis rejected or downgraded due to complete absence of quantitative metric curves (ROC, EER, mAP, Rank-1). | 4 | 5 | **20** | **CRITICAL** | External examiner points out that self-matching during enrollment is not generalization accuracy. | Execute Priority P1 benchmarking harness (`evaluate_benchmarks.py`) over LFW, VeRi-776, and CelebA-Spoof. | Present honest preliminary benchmark curves with explicit confidence intervals in Chapter 5. | AI Engineer |
| **RSK-04** | Security / AI | **Physical Presentation Attack Bypasses Anti-Spoofing:** A printed photo or screen replay held up during the live viva successfully unlocks the gate. | 4 | 4 | **16** | **HIGH** | Gate opens on a paper photo of an enrolled student during live demo. | Enforce central fail-secure policy: all face matches mandate operator review (`REVIEW_REQUIRED`). Integrate MiniFASNetV2. | Demonstrate the fail-secure modal popping up in real time, explaining that human-in-the-loop review is the intended defense. | AI Engineer / AppSec |
| **RSK-05** | Technical / Demo | **Webcam Hardware or Driver Disconnect During Viva:** Physical USB webcam fails to enumerate or crashes due to Windows camera privacy settings during the live defense. | 3 | 4 | **12** | **HIGH** | Dashboard shows black screen; OpenCV logs `Cannot open camera index 0`. | Test camera 1 hour before defense; close competing apps (Teams, Zoom); verify Windows desktop camera privacy permissions. | Immediately switch the dashboard to **Recorded Dataset Replay Mode** or play narrated 1080p backup videos (Task P3-04). | Project Lead |
| **RSK-06** | Technical / Model | **Unpinned Model Weight URL Failure:** Third-party HuggingFace/GitHub release URL breaks or returns 404, preventing system deployment on new machines. | 4 | 3 | **12** | **HIGH** | `PlateDetector` logs `urllib.error.HTTPError: 404 Not Found` on first launch. | Bundle all required model weights locally in `models/` with SHA-256 pre-verification (Task P0-03, P0-06). | Check in verified weights directly or host on a permanent institutional Google Drive / HuggingFace mirror. | Software Architect |
| **RSK-07** | Timeline / Scope | **Scope Creep (Attempting to Train Foundation Models from Scratch):** Attempting to pretrain an 86M ViT or 7B VLM burns remaining weeks without producing a working pipeline. | 3 | 4 | **12** | **HIGH** | Training scripts run for days without converging; core integration breaks. | Enforce Negative Scope Register (Roadmap §6): strictly forbid foundation training; use pretrained weights and focused transfer learning. | Abort training runs immediately; revert to validated ONNX baselines (SFace, OSNet). | Technical Lead |
| **RSK-08** | Security / Transport| **Session Hijacking via Unsecured Cookies:** Session cookies intercepted over unencrypted local networks during multi-device demonstration. | 3 | 3 | **9** | **MEDIUM** | Wireshark captures plain `ig_session` cookie on local LAN. | Implement Task P0-02 (`SameSite=Lax`, `Secure`, HMAC validation). Run demo strictly on `127.0.0.1` loopback. | Restrict demonstration to local loopback interface. | AppSec Engineer |
| **RSK-09** | Technical / Data | **License Plate OCR Format Incompatibility:** EasyOCR fails on provincial Pakistani plates due to non-standard fonts, Urdu text, or regional crests. | 3 | 3 | **9** | **MEDIUM** | OCR outputs gibberish or empty string on local vehicle test images. | Focus evaluation on English alphanumeric characters; apply adaptive thresholding and bilateral filtering. | Explain format domain shift honestly in thesis limitations; present CER curves across clean vs. degraded formats. | AI Engineer |
| **RSK-10** | Technical / DB | **SQLite Lock Contention Under High Event Volume:** Concurrent WebSocket pushes and rapid event logging cause database locked exceptions. | 2 | 3 | **6** | **MEDIUM** | Server logs `sqlite3.OperationalError: database is locked`. | Implement Task P0-05 (Enable `PRAGMA journal_mode=WAL;` and connection pooling). | Throttle event emission interval to 100 ms per camera worker. | Software Architect |

---

## 3. High-Priority Mitigation Strategy Execution

### Immediate 48-Hour Actions (Critical Risks RSK-01, RSK-02, RSK-03)
1. **RSK-02 Mitigation:** Run `git-filter-repo` on a fresh clone to expunge `face_embeddings.h5` and cached bytecode from Git history before any code is shared externally.
2. **RSK-01 Mitigation:** Draft the official 3-page *Scope Revision Note* documenting the software-only pivot and CNN/ONNX model substitution; obtain supervisor signatures.
3. **RSK-03 Mitigation:** Begin dataset retrieval for CelebA-Spoof, LFW, and VeRi-776 evaluation subsets to generate empirical accuracy curves.
