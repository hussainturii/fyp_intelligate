# INTELLIGATE: DEFENSIVE APPLICATION SECURITY & BIOMETRIC PRIVACY FORENSIC AUDIT

**Document ID:** AUD-2026-10-05  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Audit Baseline Date:** October 18, 2026  
**Auditing Standards:** OWASP Top 10 (2021), ISO/IEC 27001 Annex A, NIST SP 800-63B & GDPR Article 9 (Special Category Biometric Data)  

---

## 1. Executive Security & Privacy Overview

A rigorous defensive security assessment of the IntelliGate codebase, Git revision history, cryptographic implementations, and operational workflows was performed. The evaluation verified that the active software runtime incorporates robust defensive architectural patterns—notably **record-bound AES-256-GCM biometric encryption at rest**, **scrypt password hashing with per-user cryptographic salts**, **strict RBAC permission boundaries**, and a **fail-secure human-in-the-loop review policy**.

However, the forensic audit identified **one critical biometric exposure in Git history**, an **unpinned model download vulnerability (SEC-1 / VULN-01)**, and **missing browser session security directives**. 

In strict adherence to audit rules:
- No personal names, real secrets, or raw biometric embedding vectors are exposed in this report.
- No repository files or Git commits have been deleted or modified during this audit phase.
- An explicit, step-by-step remediation plan is formulated for execution upon stakeholder approval.

---

## 2. Vulnerability Register & Threat Analysis

| Vuln ID | Title / Vulnerability Description | Affected Component | Exploitability | Severity | CVSS v3.1 | Status |
|---|---|---|---|---|---|---|
| **VULN-01** | Unverified Third-Party Model Checkpoint Loading | `module2_vehicle/plate_detector.py:35–39, 81` | Remote / MITM | **HIGH** | 7.8 | Confirmed in Legacy |
| **VULN-02** | Plaintext Biometric Embeddings Committed to Git History | `module1_face_recognition/database/face_embeddings.h5` | Local / Repository Access | **CRITICAL** | 8.2 | Confirmed in Git |
| **VULN-03** | Missing Session Cookie Directives (`SameSite`, `Secure`) | `module8_dashboard/backend/main.py:149–153` | Network / Cross-Site | **MEDIUM** | 5.4 | Confirmed in Active |
| **VULN-04** | Plaintext AES-256 Master Biometric Key on Local Disk | `data/biometric.key` (or working directory) | Local File System Access | **HIGH** | 6.8 | Confirmed in Active |
| **VULN-05** | Potential Event Fabrication via Privileged REST Endpoints | `module8_dashboard/backend/main.py:192–210` (`POST /api/events`) | Authenticated Insider | **MEDIUM** | 4.3 | Confirmed in Active |
| **VULN-06** | Unbounded In-Memory Login Rate Limiter (Memory Exhaustion) | `module8_dashboard/backend/main.py:48–56` | Remote Denial of Service | **LOW** | 3.7 | Confirmed in Active |

---

## 3. Deep Forensic Investigation of Specific Threats

### 3.1 VULN-01: Arbitrary Code Execution via Unpinned Model Checkpoints (SEC-1)
- **Vulnerability Mechanism:**
  In `module2_vehicle/plate_detector.py`, line 35 configures a remote URL:
  `https://github.com/keremberke/awesome-yolov8-models/releases/download/v1.0.0/license_plate_detector.pt`.
  In lines 78–81, `PlateDetector._load_model()` attempts to download this file and load it via `ultralytics.YOLO(path)`.
- **Security Impact:**
  PyTorch `.pt` checkpoints utilize Python's `pickle` serialization format under the hood. Deserializing an unverified, unhashed `.pt` file from an external repository allows arbitrary code execution via the `__reduce__` exploit vector. If an attacker compromises the upstream release asset or performs a Man-in-the-Middle (MITM) substitution, malicious code executes with the full privileges of the backend process.
- **Current Runtime Status:**
  The integrated dashboard (`module8_dashboard/backend/vision.py:224`) instantiates `PlateDetector(allow_download=False)`, which bypasses the automatic network download. However, the legacy standalone module `module2_vehicle/main.py` instantiates `PlateDetector()` without this override, leaving standalone executions vulnerable.

### 3.2 VULN-02: Legacy Biometric Data Exposure in Git Commit History
- **Vulnerability Mechanism:**
  In initial commit `98df922` (*"Initial commit"*), the file `module1_face_recognition/database/face_embeddings.h5` was staged and committed into the repository history.
- **Forensic Inspection:**
  Inspection of the HDF5 metadata reveals that the file contains an enrolled face group with UUID `96ca9003-a76c-4f83-a294-946603b3f906` storing unencrypted face vectors.
- **Regulatory & Privacy Risk:**
  Under the General Data Protection Regulation (GDPR Article 9) and international biometric privacy frameworks (e.g., BIPA), facial embedding vectors constitute **Special Category Biometric Data**. Because facial embeddings cannot be modified or rotated like passwords, committing raw embeddings to Git history exposes enrolled subjects to irreversible biometric compromise (including facial reconstruction via embedding inversion attacks).
- **Audit Verification:**
  While the *active* prototype in `c909145` uses encrypted SQLite tables (`cpu_faces`) and excludes `face_embeddings.h5`, the legacy file remains tracked in Git history and can be checked out by anyone with repository read access.

### 3.3 VULN-03: Session Cookie Security Directives Missing
- **Vulnerability Mechanism:**
  In `module8_dashboard/backend/main.py`, line 149 sets the session authentication cookie:
  ```python
  response.set_cookie(
      key="ig_session",
      value=session_token,
      httponly=True,
      max_age=8 * 3600,
  )
  ```
- **Defect:**
  The cookie sets `httponly=True` (which correctly prevents JavaScript DOM access via `document.cookie`), but omits:
  1. `samesite="lax"` or `samesite="strict"`: Browsers default to sending the cookie in cross-site requests if omitted, exposing state-changing endpoints to Cross-Site Request Forgery (CSRF).
  2. `secure=True`: Cookies will be transmitted over unencrypted HTTP connections in transit.
- **Risk:**
  Low during local prototyping on `127.0.0.1`, but an immediate high vulnerability if hosted on a local network or demonstrated across devices.

### 3.4 VULN-04: Biometric Master Encryption Key Management
- **Vulnerability Mechanism:**
  In `module8_dashboard/backend/vision.py:111–121`, `FaceEngine` generates an AES-256 key and stores it as a raw binary file at `biometric.key` adjacent to the SQLite database.
- **Evaluation:**
  Setting file permissions to `0o600` is good practice on POSIX systems, but Windows NTFS access control lists (ACLs) do not enforce POSIX mode bits in the same manner. Any process running under the user's Windows user account can read `biometric.key` and decrypt the entire database of enrolled staff and visitor faces.
- **Remediation:**
  The encryption key should be supplied via an external environment variable (`INTELLIGATE_BIOMETRIC_KEY`) injected at runtime or protected by the Windows Data Protection API (DPAPI).

### 3.5 VULN-05: Event Provenance & Security Staff Event Fabrication
- **Vulnerability Mechanism:**
  The REST API exposes endpoint `POST /api/events` (`main.py:192–210`) restricted to users with the `security_staff` or `admin` role.
- **Threat Vector:**
  A malicious or negligent security operator can forge historical security events (e.g., fabricating a false `access_granted` record for an unauthorized vehicle plate or an unregistered face) by posting directly to `/api/events`.
- **System Defense Analysis:**
  Fortunately, the gate control state machine in `SoftwareRuntime` is decoupled from `/api/events`: posting an event to `/api/events` merely writes an audit log to SQLite; it **does NOT invoke `SoftwareRuntime.observe()` or trigger `SoftwareRuntime.command('OPEN')`**. However, it allows an insider to corrupt audit event provenance and falsify daily compliance reports.

---

## 4. Architectural Security Strengths

The audit verified five robust security controls already operating within the codebase:

1. **Record-Bound Additional Authenticated Data (AAD):**
   When `FaceEngine.seal(embedding, identity)` encrypts a face vector, it binds the record's UUID string as AAD:
   ```python
   nonce = os.urandom(12)
   ciphertext = self.cipher.encrypt(nonce, embedding.tobytes(), identity.encode())
   return nonce + ciphertext
   ```
   If an adversary attempts a ciphertext transplant attack (copying an authorized administrator's encrypted embedding blob into an unknown visitor's database row), the GCM authentication tag verification will fail upon decryption, raising a cryptographic exception.
2. **Scrypt Key Derivation for Credentials:**
   User passwords are not stored in plaintext or weak hashes (MD5/SHA-256). Passwords use Python `hashlib.scrypt` with a per-user cryptographically random 16-byte salt (`n=16384, r=8, p=1`), providing high resistance to GPU-accelerated dictionary and rainbow table attacks.
3. **Fail-Secure Access Policy:**
   Unvalidated biometric matches never trigger gate actuation autonomously. The policy requires explicit manual review from an authenticated operator within a 120-second TTL window.
4. **Command Journaling & Idempotency:**
   All virtual gate commands are assigned a unique UUID. Re-executing a command with the same ID returns the recorded result without duplicate motor actuation, preventing replay attacks.
5. **Origin & Session-Enforced WebSocket Streaming:**
   The WebSocket endpoint `/ws` (`main.py:640–675`) validates browser Origin against configured CORS origins and mandates an active authenticated session cookie during the connection handshake.

---

## 5. Step-by-Step Security Remediation Plan

To be executed during Phase 2 upon stakeholder approval:

### Step 1: Purge Biometric Exposure from Git History (VULN-02)
1. Ensure a full offline backup of the repository exists outside Git.
2. Execute `git-filter-repo` to permanently erase `module1_face_recognition/database/face_embeddings.h5` and any `.pyc` caches from all commits:
   ```bash
   git filter-repo --invert-paths --path module1_face_recognition/database/face_embeddings.h5
   git filter-repo --invert-paths --path-glob "**/*.pyc"
   ```
3. Update `.gitignore` to explicitly block:
   ```gitignore
   *.h5
   *.key
   *.task
   __pycache__/
   *.pyc
   ```
4. Force-push the cleansed history to origin remote.

### Step 2: Remediate Session Cookie Flags (VULN-03)
In `module8_dashboard/backend/main.py:149`, update the cookie configuration:
```python
response.set_cookie(
    key="ig_session",
    value=session_token,
    httponly=True,
    samesite="lax",
    secure=False,  # Set to True when TLS is enabled
    max_age=8 * 3600,
)
```

### Step 3: Secure Local Encryption Key Injection (VULN-04)
Modify `FaceEngine.__init__` in `vision.py` to prioritize reading the master AES-256 key from an environment variable `INTELLIGATE_BIOMETRIC_KEY`, falling back to local file storage with logged warnings.

### Step 4: Validate SHA-256 Hashes on All Model Loads (VULN-01)
Implement a centralized `ModelRegistry` class that verifies the pre-computed SHA-256 checksum of every `.pt` or `.onnx` model file before passing the path to PyTorch or OpenCV. If a hash mismatch occurs, abort execution immediately with an audit security alert.
