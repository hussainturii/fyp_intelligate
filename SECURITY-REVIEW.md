# IntelliGate Security Review

**Audit date:** 2026-10-15  
**Repository baseline:** `main` at `c90914597bba63adaf569c05f3de967676c9f47f`  
**Scope:** Read-only review of tracked application source, configuration, and dependency manifests. No destructive testing or biometric database inspection was performed.

## Findings

| ID | Severity | Affected component | Evidence | Vulnerability | Confidence |
|---|---|---|---|---|---:|
| SEC-1 | HIGH | Legacy standalone vehicle module | [`module2_vehicle/plate_detector.py`](./module2_vehicle/plate_detector.py), lines 35–39, 72–80, 215–226; constructed without an override in [`module2_vehicle/main.py`](./module2_vehicle/main.py), line 90 | The default standalone Module 2 path automatically downloads a third-party `.pt` checkpoint and passes it to YOLO without checking a pre-approved digest or signature. A substituted release artifact can reach checkpoint loading and alter or compromise the running process. | 8/10 |

### Recommended remediation

Disable automatic model downloads by default. If the standalone path must use the checkpoint, acquire it through a controlled process, verify a pinned trusted SHA-256 digest or signature before loading, and use a safe loading format/path. Add a test proving an absent or mismatched model is rejected before it is loaded.

**Scope note:** This finding concerns the legacy standalone Module 2 pipeline. The integrated dashboard uses its separate CPU pipeline and explicitly disables the plate-model downloader; it is not the affected path identified here.

## Other security observations

- No other exploitable issue was established in the reviewed HTTP/WebSocket paths. The routes reviewed require a valid session; the WebSocket checks the browser Origin when one is supplied.
- `npm ci` reported 17 advisories in the dependency tree. The production dependency audit reported three moderate React Router advisories; no attacker-controlled redirect or server-side-rendering path matching those advisories was found in this application. Remaining advisories are in build/development dependencies and did not establish an exploitable application path. Review compatible dependency updates and rerun the audit.
- The repository tracks `module1_face_recognition/database/face_embeddings.h5`. Its contents were intentionally not inspected. The README notes that legacy HDF5 biometric data is not automatically migrated or encrypted; verify that this committed file contains no real personal biometric data before redistributing the repository. If it does, treat that data as exposed and remove it from repository history as well as current files.
- The software is documented as a local prototype without production TLS or production key custody. Do not expose it to an untrusted network or use it to control physical gate hardware.

## Verification record

- Frontend `npm ci` and production `npm run build` completed; the build emitted a large-chunk advisory.
- Backend tests and broader project limitations are recorded in the audit response. No security changes were made as part of this review.
- This review is static and does not constitute a production penetration test or a review of biometric data contents.
