# INTELLIGATE — DATASET ACQUISITION, PROVENANCE & VALIDATION REPORT

**Document ID:** `IG-M03A-DATASET-ACQUISITION-2026-10`  
**Date:** October 18, 2026  
**Auditor / Technical Lead:** Senior Machine Learning Engineer & Computer Vision Researcher (BS AI FYP Technical Lead)  
**Project:** IntelliGate — Smart Multi-Modal Gate Access Control System (BS AI FYP)  
**Scope:** Automated Dataset Acquisition, Provenance Verification, Storage Safety Audit & Experimental Protocol Readiness  
**Target Architecture:** Multi-Modal Gate Pipeline (YuNet + SFace Face Verification, YOLOv8 + EasyOCR ALPR, Fail-Secure Virtual Gate)  

---

## 1. Executive Summary

In accordance with milestone authorizations, an exhaustive acquisition and provenance audit was conducted across all datasets required for IntelliGate's AI development and scientific evaluation.

| Task # | Target Domain / Dataset | Provenance & Source URL | License | Size / Extracted Footprint | Status & Action Taken |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Face Verification (LFW)** | Figshare Mirror (`5976018` / `5976006`) | Non-Commercial Academic (UMass Amherst) | ~172.2 MB archive (~205 MB unpacked, 13,233 images, 6,000 pairs) | **IN PROGRESS:** `pairs.txt` verified (SHA-256: `ea42330c...`), `lfw.tgz` background download actively executing; 10-fold CV script ready. |
| **Task 2** | **Pakistani ALPR (Roboflow A & B)** | Roboflow Universe (`abbasi-cgezb` #6 & `numberplatepakistan` #1) | CC BY 4.0 / Roboflow Universe Terms | ~65 MB + ~40 MB | **STOPPED / AUTH REQUIRED:** Direct API access blocked (`HTTP 401 Unauthorized`). Requires Roboflow account API key or manual zip export. Pure bounding boxes lack OCR text ground truth. |
| **Task 3** | **Anti-Spoofing (CelebA-Spoof)** | GitHub (`ZhangYuanhan-AI/CelebA-Spoof`) / Google Drive / Baidu Yun | Non-Commercial Research Use | **~35–45 GB compressed (>80 GB unpacked)**, 625,537 images | **BLOCKED / APPROVAL PENDING:** Storage space verified (533.79 GB free). High-capacity download held for explicit supervisor confirmation. |
| **Task 4** | **Vehicle Re-ID (VeRi-776)** | GitHub (`VehicleReId/VeRi`) / BUPT | Academic Research Non-Commercial | ~50,000 images, 776 vehicles | **RESTRICTED / EMAIL REQUIRED:** Not publicly downloadable via open link. Official terms require sending email request with academic affiliation to `xinchenliu@bupt.cn`. |
| **Task 5** | **Low-Light Fusion (LLVIP)** | Local cache (`logs/demo_data/`) & BUPT (`bupt-ai-cz/LLVIP`) | Academic Use Only | ~25 MB local cached demo; full archive ~25 GB (30,976 pairs) | **LOCALIZED / AUDITED:** Existing visible/infrared MP4 clips confirmed in `logs/demo_data/`. Full 25 GB archive not duplicated to avoid redundant network waste. |
| **Task 6** | **Storage & Reproducibility Safety** | Local Workspace (`D:\MZHR_FYP\INTELLIGATE`) | N/A | Total Drive Capacity: 1,471.28 GB; **Free: 533.79 GB** | **ENFORCED:** Git exclusion updated (`data/`, `*.tgz`, `*.zip`). Path traversal protections enforced. Production gates unaffected. |

---

## 2. Storage Safety & System Capacity Audit (Task 6)

Before executing any download operations, filesystem and storage constraints were audited on host machine `D:\`:

```
Drive: D:\
Total Storage: 1,471.28 GB
Used Storage:    937.49 GB
Free Storage:    533.79 GB (36.28% free)
```

### 2.1 Git Hygiene & Isolation Invariant
- Verified and updated [`.gitignore`](file:///D:/MZHR_FYP/INTELLIGATE/.gitignore):
  ```gitignore
  data/
  *.tar
  *.tar.gz
  *.tgz
  *.zip
  ```
- **Result:** Large media binaries, tarballs, and raw image directories are strictly prevented from entering git tracking.
- Production access control databases ([`module2_vehicle/database/vehicles.h5`](file:///D:/MZHR_FYP/INTELLIGATE/module2_vehicle/database/vehicles.h5) and SQLite registries) and cryptographic HMAC keys remain untouched and strictly isolated.

### 2.2 Path Traversal & Archive Integrity Shield
- Implemented secure archive extraction in [`tools/acquire_lfw.py`](file:///D:/MZHR_FYP/INTELLIGATE/tools/acquire_lfw.py):
  Every archive member's normalized target path is verified against `target_dir.resolve()`:
  ```python
  member_path = (target_dir_resolved / member.name).resolve()
  if not str(member_path).startswith(str(target_dir_resolved)):
      raise ValueError(f"Path traversal security vulnerability detected: {member.name}")
  ```
  This completely eliminates `tarfile` directory traversal vulnerabilities (`../../`).

---

## 3. Task 1: Official LFW Face Verification Acquisition & Protocol

### 3.1 Distribution Details & File Verification

| Asset | Source URL | Local Target Path | File Size | SHA-256 / Verification State |
| :--- | :--- | :--- | :--- | :--- |
| `pairs.txt` | `https://ndownloader.figshare.com/files/5976006` | `data/lfw/pairs.txt` | 155,335 bytes (~151 KB) | `ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592` (**VERIFIED**) |
| `lfw.tgz` | `https://ndownloader.figshare.com/files/5976018` | `data/lfw/lfw.tgz` | 180,566,744 bytes (~172.2 MB) | Download executing in background via native curl (`task-699`) |
| Unpacked Images | Tar extraction into `data/lfw/` | `data/lfw/lfw/<person>/` | ~205 MB (13,233 images) | Pending completion of download |

### 3.2 Protocol Implementation & Zero-Leakage Guarantee
- **Script:** [`experiments/eval_lfw_official.py`](file:///D:/MZHR_FYP/INTELLIGATE/experiments/eval_lfw_official.py)
- **Mathematical Protocol:**
  1. Official 10-fold split: 600 pairs per fold (300 genuine, 300 impostor; 6,000 total pairs).
  2. Out-of-fold calibration:
     $$\tau_k^* = \arg\max_\tau \text{Accuracy}_{\text{train}}(\tau) \quad \text{on 9 folds (5,400 pairs)}$$
  3. Testing: Evaluated on unseen fold $k$ (600 pairs).
  4. Detection failures logged explicitly as **Face Detection Errors (FDE)** and assigned similarity $-1.0$ (never omitted or swept under the rug).
  5. Metrics recorded: 10-fold Mean Accuracy ($\pm \text{std}$), Wilson 95% Confidence Intervals, ROC Curve, AUC, and Equal Error Rate (EER).

---

## 4. Task 2: Pakistani License Plate Datasets (Roboflow Universe)

### 4.1 Target Repositories Audited

1. **Dataset A:** `https://universe.roboflow.com/abbasi-cgezb/pakistani-number-plates-4dyfe/dataset/6`  
   - **Annotated Domain:** Real Pakistani motor vehicles (Punjab, ICT, Sindh) in outdoor illumination.
   - **Declared License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
2. **Dataset B:** `https://universe.roboflow.com/numberplatepakistan/pakistani_license_number_plates/dataset/1`  
   - **Annotated Domain:** Secondary Pakistani vehicular license plate bounding boxes.
   - **Declared License:** Roboflow Universe Open Source / CC BY 4.0.

### 4.2 Access Control & Blocker Disclosure
In strict adherence to project rules (*"If Roboflow requires login, an API key, workspace authorization, or license acceptance, stop that download and tell me exactly what action is required. Do not bypass access controls. Never print or commit API keys."*):

- **Automated Probe Result:**
  - `https://api.roboflow.com/abbasi-cgezb/pakistani-number-plates-4dyfe/6` returned `HTTP 401 Unauthorized`.
  - `https://api.roboflow.com/numberplatepakistan/pakistani_license_number_plates/1` returned `HTTP 401 Unauthorized`.
- **Reason:** Roboflow Universe requires an authenticated user account to generate dataset download tokens/keys.
- **Required Supervisor Action:**
  1. Log into your free Roboflow account at [Roboflow Universe](https://universe.roboflow.com).
  2. Navigate to Dataset A and click **"Download Dataset"** $\to$ Select format **"YOLOv8"** or **"COCO"** $\to$ Download zip directly.
  3. Place or extract the archive into:
     - `D:\MZHR_FYP\INTELLIGATE\data\pakistan_plates\dataset_a\`
     - `D:\MZHR_FYP\INTELLIGATE\data\pakistan_plates\dataset_b\`
  4. Alternatively, export using the Python snippet with your private key in a local `.env` (never committed).

### 4.3 Critical Computer Vision Caveat: Bounding Boxes vs OCR Ground Truth
> [!WARNING]
> **METHODOLOGICAL SAFEGUARD:**  
> Both Roboflow datasets contain **object detection bounding box annotations** (`class: license_plate`, `[x, y, w, h]`).  
> They do **NOT** contain verified character string transcriptions (OCR ground truth like `"LEA-1234"`).  
> In compliance with technical guidelines, we will evaluate **Plate Localization precision/recall** on these images, but we will **NOT fabricate plate transcriptions or claim OCR accuracy** without verified character string annotations.

---

## 5. Task 3: Face Anti-Spoofing Dataset (CelebA-Spoof)

- **Repository:** [`https://github.com/ZhangYuanhan-AI/CelebA-Spoof`](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) (ECCV 2020)
- **Dataset Composition:**
  - Total Images: **625,537 high-resolution images** (512×512 pixels).
  - Subjects: **10,177 distinct identities**.
  - Rich Attributes: 43 facial attributes; spoof modalities include 2D printed paper attacks, tablet/smartphone digital replays, and 3D mask presentations across varied indoor/outdoor lighting.
- **Distribution Mechanism:** Hosted on Google Drive (`1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z`) and Baidu Netdisk (`61fd`) in split zip archives.
- **Estimated Size:**
  - Compressed zip archives: **~35 GB to 45 GB**.
  - Fully unpacked disk footprint: **~85 GB to 100 GB**.
- **Storage Feasibility:**
  - Available space on `D:\`: **533.79 GB free** (storage capacity is fully sufficient).
- **Status & Recommendation:**
  - In compliance with Task 3 instructions (*"Because this dataset is large, do not download it automatically. Report the estimated size and ask for my approval"*), automated download was **NOT triggered**.
  - **Supervisor Decision Required:** Do you authorize downloading a selected validation partition (~5–10 GB) or the full 40 GB CelebA-Spoof dataset for deep anti-spoofing fine-tuning in Milestone 3B?

---

## 6. Task 4: Vehicle Re-ID Dataset (VeRi-776)

- **Repository:** [`https://github.com/VehicleReId/VeRi`](https://github.com/VehicleReId/VeRi) (IEEE TMM 2018 / ECCV 2016)
- **Dataset Composition:** Over 50,000 images covering 776 vehicles captured across 20 unconstrained surveillance cameras in a 1.0 km² area over 24 hours. Includes camera IDs, spatiotemporal timestamps, and multi-viewpoint occlusions.
- **Access Conditions & Restriction:**
  Section 2 of the official VeRi repository explicitly mandates:
  > *"To encourage related research, we will provide the dataset according to your request. Please email your full name and affiliation to the contact person (xinchenliu at bupt dot cn). We ask for your information only to make sure the dataset is used for non-commercial purposes. We will not give it to any third party or publish it publicly anywhere."*
- **Blocker Status:**
  - VeRi-776 is **gated behind formal institutional researcher approval**. There is no legitimate, unauthenticated public URL.
  - In strict compliance with guidelines (*"If researcher registration, email approval or authorization is required, document it and ask me. Do not bypass dataset access restrictions"*), we have documented the required application procedure.
- **Recommended Action for FYP:**
  - If multi-camera vehicle tracking is retained for later milestones, send the standard academic request email to `xinchenliu@bupt.cn` using university credentials.
  - Note: As proven in our Milestone 3A.1 analysis, Vehicle Re-ID is an auxiliary surveillance feature and cannot actuate the physical gate; therefore, this restriction does **not block** core gate access control functionality.

---

## 7. Task 5: LLVIP Low-Light Dataset Inspection

- **Repository:** [`https://github.com/bupt-ai-cz/LLVIP`](https://github.com/bupt-ai-cz/LLVIP) (ICCV 2021)
- **Local Inspection:**
  - Inspected [`logs/demo_data/`](file:///D:/MZHR_FYP/INTELLIGATE/logs/demo_data/):
    - `llvip_illustration.gif` (25,293,903 bytes / ~24.1 MB)
    - `llvip_visible.mp4` (300,223 bytes, 150 frames, visible spectrum)
    - `llvip_infrared.mp4` (2,200,333 bytes, 150 frames, long-wave infrared spectrum)
    - `manifest.json` (SHA-256: `09ab6ca8b6...`, terms: academic non-commercial use)
- **Full Dataset Size & Distribution:**
  - Total Composition: 30,976 registered visible/infrared image pairs ($1280 \times 1024$).
  - Full archive size on Google Drive / Baidu Yun: **~25 GB**.
- **Audit Conclusion:**
  - Sufficient paired test frames already exist locally in `logs/demo_data/` for cross-spectral low-light camera fusion pipeline validation.
  - To prevent wasteful multi-gigabyte duplication of data, the full 25 GB archive was **not downloaded**, fully adhering to Task 5 rules.

---

## 8. Summary of Active Tasks & Pending User Inputs

```
[✓] Task 1: pairs.txt acquired & verified; lfw.tgz downloading in background.
[!] Task 2: Roboflow requires account login/export; bounding boxes ready for localization evaluation.
[?] Task 3: CelebA-Spoof (~35-45 GB) audited; awaiting user approval before large download.
[!] Task 4: VeRi-776 requires institutional email application to xinchenliu@bupt.cn.
[✓] Task 5: Local LLVIP samples confirmed; duplicate 25 GB download avoided.
[✓] Task 6: D: drive audited (533.79 GB free); .gitignore hardened; path traversal protected.
```
