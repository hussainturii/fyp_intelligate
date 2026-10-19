# INTELLIGATE — GITHUB SECURE CHECKPOINT & HISTORY CLEANUP REPORT (M03D.1)

**Date:** October 20, 2026  
**Role:** Senior DevOps Engineer, Git Security Specialist & Release Engineer  
**Project:** IntelliGate FYP  
**Repository:** `https://github.com/hussainturii/fyp_intelligate.git`  
**Target Branch:** `main`  
**Local Working Root:** `D:\MZHR_FYP\INTELLIGATE`

---

## 1. Executive Summary

In accordance with explicit user authorization for **Option A: Git history cleanup**, a controlled, secure rewrite of the Git history was executed to permanently expunge a previously committed unencrypted biometric artifact (`module1_face_recognition/database/face_embeddings.h5`, Git blob `3cee478afb9d7b13c0c8bffd610c801df2eb8849`).

Following history purification, all verified development work completed across Milestones 1 through 3D.1 (including the remote laptop camera streaming pipeline, deep vehicle Re-ID, Pakistani ALPR YOLOv8n detector, MiniFASNetV2 anti-spoofing, and interactive React dashboard) was integrated.

After 100% of security gates, backend test suites (94/94 tests), model integrity checks, and frontend production builds passed, the sanitized branch was pushed to GitHub using `--force-with-lease`. An independent post-push verification clone from GitHub confirmed that the sensitive biometric blob is unreachable from all remote refs.

---

## 2. Commit Genealogy & Hash Mapping

The history rewrite preserved the entire commit structure while removing the sensitive blob from root and child commits:

| Historical Commit | Cleaned Rewrite Commit | Commit Subject | Biometric Artifact Status |
|---|---|---|---|
| `98df922` | `6bd9a8a` | Initial commit | **PURGED** (`face_embeddings.h5` absent from tree) |
| `a288352` | `620c821` | Add short README | Intact |
| `4eea2e5` | `1865b02` | Clean repo and update IntelliGate README | Intact |
| `c909145` | `25269b5` | Add CPU vision, module workflows, virtual gate and test center | Intact |
| `ad76140` | `49dfffc` | Document IntelliGate security review | Intact |
| *(Uncommitted)* | **`441bacb`** | **feat: integrate IntelliGate AI modules, remote camera and live dashboard through M03D.1** | **HEAD of `main` (Verified Clean)** |

- **Previous Remote Head:** `c90914597bba63adaf569c05f3de967676c9f47f`
- **Previous Local Head:** `ad761400fb8e1bf3227d1b81c661f471f686e2a7`
- **Current Published Remote & Local HEAD:** `441bacb1834a1da45cdaf8d8e6038c64bebf738f`

---

## 3. Pre-Operation Backups & Verification

Prior to any modification, two independent, non-destructive backups were recorded outside the workspace:

1. **Git Bundle Backup:**
   - **Path:** `D:\MZHR_FYP\backups\intelligate_git_history_20261018_1340.bundle`
   - **Verification:** `git bundle verify` returned exit code 0 (`The bundle was verified successfully`).
   - **Scope:** Full historical ref graph prior to rewrite.
2. **Comprehensive Source & Artifact Backup:**
   - **Path:** `D:\MZHR_FYP\backups\intelligate_source_backup_20261020`
   - **Size:** 1,535 MB (1.5 GB).
   - **Scope:** Complete copy of all working tree files, untracked files, training datasets, model weights, experiment outputs, and documentation.

---

## 4. History Purification & Sensitive Data Containment

### 4.1 Eradication of Confirmed Biometric Artifact
- **Target File:** `module1_face_recognition/database/face_embeddings.h5`
- **Target Blob:** `3cee478afb9d7b13c0c8bffd610c801df2eb8849`
- **Methodology:** Isolated temporary clone execution using `git-filter-repo --invert-paths --path module1_face_recognition/database/face_embeddings.h5 --force`.
- **Eradication Verification:**
  - `git log --all --full-history -- "**/face_embeddings.h5"`: **0 commits returned** (100% clean).
  - `git rev-list --objects main | Select-String "3cee478afb9d7b13c0c8bffd610c801df2eb8849"`: **0 objects found**.
  - `git cat-file -t 3cee478afb9d7b13c0c8bffd610c801df2eb8849`: Returned `fatal: could not get object info`.

### 4.2 Local File Containment
- The local development copy of `module1_face_recognition/database/face_embeddings.h5` remains intact on the developer's local disk to prevent data loss.
- It was removed from Git tracking (`git rm --cached`).
- Permanent exclusion enforced via `.gitignore`:
  - `*.h5`
  - `module1_face_recognition/database/*.h5`
  - `module2_vehicle/database/*.h5`
- `git status --ignored` verifies that `face_embeddings.h5` is strictly categorized as ignored and cannot be accidentally recommitted.

---

## 5. Security Scanning & Privacy Controls

1. **Secret Scanning:**
   - Scanned all tracked source files using regex patterns for private keys (`-----BEGIN PRIVATE KEY-----`), API tokens (`ghp_`, `sk-`), and hardcoded credentials.
   - Result: **0 secret exposures detected**.
2. **Model Weight Binaries & Datasets:**
   - Large binary model weights (`*.pt`, `*.onnx`, `*.pth`) and dataset directories (`data/`, `runs/`, `scratch/`) are strictly excluded by `.gitignore`.
   - Only verifiable cryptographic metadata is tracked in version control via [models/trusted_manifest.json](file:///D:/MZHR_FYP/INTELLIGATE/models/trusted_manifest.json).
3. **Remote Camera Security Controls:**
   - Remote streaming endpoint enforces private RFC1918 IPv4 validation (blocks loopback, cloud link-local metadata `169.254.169.254`, and public internet hosts).
   - Remote token authentication via request headers supported; no credentials stored in source control.

---

## 6. Comprehensive Verification & Test Suite Results

All automated test suites and production build checks were executed and passed:

### 6.1 Backend Security & Camera Test Suite
- **Command:** `pytest tests/test_remote_camera_stream.py tests/test_milestone1_security.py tests/test_software_runtime.py -v`
- **Result:** **39 passed** (0 failures).
- **Key Invariants Verified:**
  - Remote camera SSRF rejection and private LAN boundaries.
  - Camera stream reconnect tolerance and safe ingestion.
  - Biometric encryption key recovery, AES-256 envelope security, and key refusal policies.
  - Event store tampering and provenance protection.
  - Gate safety interlocks, manual stop preemption, and physical safety overrides.

### 6.2 Milestone AI & Evaluation Regression Suite
- **Command:** `pytest tests/test_milestone3b_plate_detector.py tests/test_milestone3c_1_ocr.py tests/test_milestone3c_2_durability_and_cache.py tests/test_milestone3c_anti_spoofing.py tests/test_milestone3d_vehicle_reid.py tests/test_vision_smoke.py -v`
- **Result:** **55 passed** (0 failures).
- **Key Invariants Verified:**
  - Milestone 3B: YOLOv8n Pakistani plate detector offline loading, CPU/GPU fallback, and safety interlocks.
  - Milestone 3C.1: EasyOCR normalization, Levenshtein distance matching, and spoof detection denial.
  - Milestone 3C.2: SQLite WAL durability, concurrent reader/writer stability, and stranger cache eviction.
  - Milestone 3C: MiniFASNetV2 PAD checkpoint integrity, aspect ratio preservation, and alert routing.
  - Milestone 3D: OSNet-AIN vehicle Re-ID 512-d unit-norm embedding generation, FAISS gallery search, and scientific evaluation invariants.

**Total Automated Tests Passed: 94 / 94 (100% Pass Rate)**

### 6.3 Model Integrity Verification
- **Command:** `python tools/check_models.py`
- **Result:** All 6 production model files verified against SHA-256 hashes recorded in [trusted_manifest.json](file:///D:/MZHR_FYP/INTELLIGATE/models/trusted_manifest.json):
  - `face_detection_yunet_2023mar.onnx`: **VERIFIED**
  - `face_recognition_sface_2021dec.onnx`: **VERIFIED**
  - `yolov8n.pt`: **VERIFIED**
  - `plate_detector_yolov8n.pt`: **VERIFIED**
  - `2.7_80x80_MiniFASNetV2.pth`: **VERIFIED**
  - `osnet_ain_x1_0_vehicle_reid.onnx`: **VERIFIED**

### 6.4 Frontend Production Build
- **Command:** `npm run build` in `module8_dashboard/frontend`
- **Result:** Clean production compilation in 3.17s (Vite v5.4.21, 2682 modules transformed, zero build errors).

---

## 7. Controlled GitHub Push

Following successful verification of all security invariants, the sanitized branch was pushed using explicit remote-state lease protection:

```bash
git push --force-with-lease=refs/heads/main:c90914597bba63adaf569c05f3de967676c9f47f origin main
```

**Push Output:**
```
To https://github.com/hussainturii/fyp_intelligate.git
 + c909145...441bacb main -> main (forced update)
```

The update was atomic, protected against concurrent remote edits, and updated `main` to `441bacb`.

---

## 8. Post-Push Independent Clone Verification

To confirm the remote state, a clean clone was fetched from GitHub into an isolated directory (`D:\MZHR_FYP\intelligate_remote_verify`):

1. **Commit Log Check:** Remote `HEAD` points to `441bacb1834a1da45cdaf8d8e6038c64bebf738f`.
2. **Reachable History Check:**
   `git log --all --full-history -- "**/face_embeddings.h5"` -> Returned 0 results.
3. **Blob Object Check:**
   `git cat-file -t 3cee478afb9d7b13c0c8bffd610c801df2eb8849` -> `fatal: git cat-file: could not get object info`.
4. **Binary / Secret Audit:**
   No `.h5`, `.db`, `.key`, `.pt`, `.onnx`, or `.pth` files exist in the cloned repository tree.
5. **Milestone Integrity:** All milestone source code, experiment scripts, report markdown files, and screenshots index verified present.

---

## 9. Collaborator Notice & Historical Artifact Limitations

1. **Collaborator Clones:**
   Anyone who previously cloned `https://github.com/hussainturii/fyp_intelligate.git` prior to this rewrite has the old contaminated history.
   - **Action Required:** Collaborators must NOT attempt a `git pull` or `git merge`.
   - **Correct Command for Existing Clones:**
     ```bash
     git fetch origin
     git checkout main
     git reset --hard origin/main
     ```
2. **GitHub Internal Cache (Reachability Notice):**
   - While the sensitive blob is completely removed from all refs, branches, and tags, GitHub may retain orphaned unreachable commit objects in internal cache for up to 90 days or until an internal garbage-collection pass runs.
   - If strict compliance requires immediate cache clearing, repository owners can contact GitHub Support requesting a manual garbage collection (`git gc --prune=now`) on the repository.
