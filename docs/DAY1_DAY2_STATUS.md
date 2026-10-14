# Day 1 / Day 2 software delivery

Implemented and checked 12 October 2026 against the software-only scope. Original model-specific SRS compliance is not implied.

Later on 12 October the user explicitly requested webcam support. That request supersedes the earlier camera-hardware exclusion: live webcam/replay capture now runs independently of inference, faces/vehicles use track caches and appearance-scoped event suppression, and camera-frame enrolment is available. Gate hardware remains excluded. See `MODULE_TEST_GUIDE.md` for current test steps.

## Implemented

- No automatic webcam startup; uploaded/curated recorded sources and operator-started webcam inputs.
- CPU YuNet detection and SFace recognition with a versioned gallery separate from legacy ViT records.
- AES-256-GCM storage for the new face/stranger embeddings, with record-bound authentication and a local key excluded from Git.
- Administrator image enrolment/removal and plate/optional vehicle-image registration/removal.
- YOLOv8n vehicle/person detection, existing EasyOCR, contour plate fallback and colour-histogram appearance candidates.
- Central access policy: appearance matches and unvalidated face liveness request review; only exact registered plates with sufficient OCR confidence may automatically open the simulated gate.
- Persistent virtual gate, transition ticks, auto-close, STOP pre-emption, obstruction/lockdown interlocks, expiring commands/reviews and idempotent command IDs.
- Encrypted unknown-face candidate matching and distinct replay-session visits; three visits in 24 hours escalate once per rolling alert window. Continuous frames do not count as visits.
- Basic IoU tracking, configurable normalized queue/gate ROIs, congestion thresholds and continuous-track dwell alerts.
- Luminance profile events; CLAHE low-light enhancement; genuine recorded infrared replay with explicit modality warning.
- Evidence-linked template reports, bounded evidence preview, JSON download and documented keyword search.
- Local administrator bootstrap, scrypt password hashes, HttpOnly expiring sessions, logout, viewer/staff/admin permissions and login throttling.
- Authenticated WebSocket and MJPEG preview; visible simulation/replay/module-limit labels; CPU health and inference timing.

## Real measurements

Eight sampled frames from the existing local `video test.mp4`, resized to 640x360; warmed inference on this i7-1165G7 CPU host:

| Stage | Median | p95 | Mean inference FPS |
| --- | --- | --- | --- |
| YuNet face detection | 54.82 ms | 60.06 ms | 18.06 |
| YOLO scene detection / basic tracking | 151.09 ms | 160.39 ms | 6.67 |
| CLAHE low-light enhancement | 12.67 ms | 16.06 ms | 78.75 |

The sampled clip frames contained zero detected faces; the face row measures detection throughput, not recognition or liveness. A separate official OpenCV static sample produced one detected face and a successful enrolment/self-match smoke test (approximately 170 ms combined). Same-image matching is not an accuracy evaluation.

Vehicle pipeline startup, including already-cached EasyOCR weights, took approximately 9.5 seconds. One first vehicle frame took approximately 0.79 seconds and produced a real unknown-vehicle observation. Warmed OCR accuracy and whole-pipeline p95 remain unmeasured. The existing plate-model URL returned 404, so plate localization used the disclosed OpenCV contour fallback.

The official LLVIP illustration was split into visible/infrared replays. Eight sampled paired frames gave median YOLO person inference times of 141.53 ms visible, 135.53 ms enhanced visible and 135.13 ms infrared. Visible detection counts were `[0,0,0,0,0,0,0,1]`, enhanced visible `[0,0,0,0,0,0,0,0]`, infrared `[1,1,1,2,1,1,0,0]`. These are unannotated illustration samples, not a held-out accuracy benchmark. They establish neither superior recall nor enhancement effectiveness.

Reproduction commands and full JSON outputs are described in the README. Measured values vary with load. The 15 FPS full-pipeline SRS target has not been met.

## Verification

- Original Day 1/2 backend suite: 27 passing tests. The live/tracking extension adds tests for four concurrent face identities, event suppression, leave/return, vehicle caching and non-blocking capture (33 passing tests at the first extension check).
- Backend tests cover role denial, forged biometric claims, login throttling, WebSocket sessions, upload playback, command idempotency, STOP precedence, obstruction sources, lockdown, restart persistence, expired reviews, visitor deduplication, bounded report previews and ROI tracking.
- Real-model smoke tests cover encrypted embedding integrity, gallery reload/removal and enrolment -> recognition candidate -> operator approval -> virtual opening.
- Frontend production build passes. An advisory bundle-size warning remains.
- Browser opens the local initial-administrator setup screen.

## Explicit remaining limitations

- No validated anti-spoof model is connected to the new CPU face path. Every face match requires manual approval. Existing legacy blink/texture/depth code remains available for future evaluation, without a claim that it rejects photo/replay/mask attacks.
- Vehicle appearance is a colour-histogram baseline, not trained ViT/OSNet Re-ID. It cannot authorize entry. Plate OCR accuracy remains unmeasured.
- Tracking is a small IoU baseline, not ByteTrack. Counts and dwell rules are not SSM-based prediction. Tailgating/forced-entry recognition is not validated.
- Reports are event templates, not VLM captions. Search is a limited keyword parser, not unrestricted natural-language reasoning.
- LLVIP illustration clips demonstrate real recorded infrared inputs. Annotated dataset test splits, fine-tuning and calibrated thermal temperatures remain outside this delivery.
- SQLite/HDF5 remains; no PostgreSQL migration, production TLS deployment, immutable audit anchoring or production reliability claim.
- Biometrics key and data reside on this local development host; production key custody/backups and formal privacy review remain future work.
- Software safety interlocks do not establish physical actuator safety or 200 ms sensor-to-actuator compliance.

## Sources

- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [LLVIP official dataset/project](https://github.com/bupt-ai-cz/LLVIP) and linked use terms. Illustration provenance and file hashes are saved locally in `logs/demo_data/manifest.json`.

These samples must not be used for identity recognition on LLVIP; the source terms prohibit identifying people in that dataset.
