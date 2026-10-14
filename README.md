# IntelliGate - CPU software prototype

An integrated live-webcam and recorded-video security dashboard built from the existing FYP codebase. Physical gate sensors and actuators are excluded. All gate actions are simulated. Webcams open only on explicit operator request.

The current delivery includes multi-face tracking with cached identities, face/vehicle enrolment, CPU recognition candidates, operator approvals, a persistent virtual gate, stranger visit rules, queue/obstruction/dwell analytics, low-light enhancement, real infrared replay, event history and evidence reports. See the [module testing guide](docs/MODULE_TEST_GUIDE.md), [delivery status and measured limitations](docs/DAY1_DAY2_STATUS.md) and [initial software-only scope](docs/SOFTWARE_ONLY_DELIVERY_PLAN.md).

The [Test Center](docs/TEST_CENTER.md) adds isolated component checks, current-camera diagnosis and image/video tests with expected names/counts and downloadable evidence. The 14 October fix prevents a delayed first webcam frame from silently killing recognition while the preview continues. Restart the backend and refresh the dashboard to load it.

## Quick start

Python 3.11 and Node.js are verified on the development host. A GPU is not needed.

Install CPU PyTorch first, then runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt -c constraints-cpu.txt
python tools/download_face_models.py
python tools/prepare_demo_data.py
```

`constraints-cpu.txt` records direct package versions tested on this Windows machine; it is not a complete transitive lockfile or a cross-platform compatibility guarantee. The legacy `requirements.txt` additionally supports the original research pipelines.

Build the dashboard and start the backend from the repository root:

```powershell
Push-Location module8_dashboard/frontend
npm ci
npm run build
Pop-Location
python -m uvicorn module8_dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Create your first local administrator account using a password of at least ten characters. There is no default password. Subsequent users can be created on the Operations page as administrator, security staff or viewer.

`start_intelligate.bat` starts the backend on loopback using the already-built frontend. For frontend development, run `npm run dev` in the frontend folder and open http://localhost:5173 with the backend running. Use the same hostname consistently so cookie sessions work.

## Demonstration walkthrough

1. On **Face Database**, enrol a clear, permitted single-face image. CPU SFace records are separate from legacy ViT records. New face/stranger embeddings are encrypted with AES-256-GCM.
2. On **Vehicle DB**, register a plate and owner. An optional cropped vehicle image provides a colour-histogram appearance baseline.
3. On **Camera / Replay**, choose face mode and press Open live camera (index 0), or upload/select a recorded clip. Capture/preview and inference use separate threads with a single pending frame. Only one source runs at a time; recordings stop at EOF.
4. Face candidates and appearance matches appear in **Operations -> Access review**. An authorized operator can approve/deny within two minutes. Unvalidated liveness cannot auto-open the virtual gate.
5. Use **scene** mode for person/vehicle detection, ROI counts, obstruction and dwell rules. Configure the blue queue ROI and red virtual-gate-path ROI on Operations.
6. Demonstrate OPEN/CLOSE/STOP/LOCKDOWN/RESET. Obstruction prevents motion; STOP cancels pending transitions. Active motion becomes STOPPED after backend restart.
7. Inspect persisted events and alerts. Generate/download the last-24-hour evidence report or search supported terms such as `face today`, `vehicle`, `queue`, `spoof` and `obstruction`.

The static OpenCV face sample is for recognition smoke tests only. It is a repeated photograph, so it cannot demonstrate liveness. LLVIP samples are split from the official paired illustration and contain genuine infrared imagery; they are unannotated visualization data, not a test-set accuracy benchmark. Identity recognition on LLVIP is disabled for curated samples.

## CPU and model limits

- YuNet + SFace provide lightweight pretrained detection/recognition. A provisional 0.45 cosine threshold is used; local identity-disjoint calibration is still needed.
- Known faces get green name boxes, unknown faces red boxes, and each visible face has a separate track. Matching/OCR is cached for continuous tracks. A two-second absence ends an appearance. Ambiguous tracking or gallery updates can trigger a fresh match; continuously present identities do not receive timed repeat events.
- No validated anti-spoof model is connected to this new CPU face path. Manual approval is required for every face match, including static/replay inputs.
- YOLOv8n detects people/vehicles. EasyOCR reads plate crops. The original specialized plate-model URL currently returns 404; plate localization falls back to OpenCV contours.
- Vehicle appearance uses colour histograms, not trained ViT/OSNet Re-ID. Appearance candidates always require review.
- Scene tracking is a basic IoU baseline. Queue/dwell rules do not establish behavioural intent or SSM prediction.
- CLAHE is classical enhancement, not a trained low-light network. Recorded infrared uses an RGB-trained detector and its accuracy is unvalidated.
- Reports use stored-event templates and search uses a limited keyword parser. These are not LLaVA/VLM inference.
- Playback FPS and inference sampling differ. Sampled scene inference measured about 6.7 FPS on this host; the full-pipeline 15 FPS SRS target is unmet.
- SQLite/HDF5 remains. This is a local software demonstration, without physical safety, PostgreSQL, production TLS, immutable audit or 99.5% uptime claims.

The dashboard status panel discloses these implementation choices for all ten SRS modules. The original modules 1 and 2 remain in the repository for research; the integrated CPU dashboard uses `module8_dashboard/backend/vision.py` and a separate new gallery.

## Tests and benchmarks

```powershell
python -m pytest tests -q
python tools/run_project_checks.py
python tools/benchmark_cpu.py --video "D:\fyp\nre\video test.mp4" --frames 8
python tools/benchmark_llvip.py
```

Benchmark JSON is saved under `logs/benchmarks/`. Timings measure warmed sampled inference, not annotated accuracy or end-to-end actuator latency. Optional model smoke tests require the face-model and demo-data preparation commands above.

## Local data and recovery

- Events/accounts/sessions/reviews/device state: `logs/events.db`.
- New vehicle gallery: `logs/cpu_vehicles.h5`.
- Encrypted face/stranger records: tables inside the event database.
- Face encryption key: `logs/biometric.key`. Preserve this key with database backups; losing it prevents decryption. Never commit it.
- Clips, sample provenance and model hashes: `logs/video_uploads`, `logs/demo_data/manifest.json` and `models/download_manifest.json`.
- `IG_EVENT_DB` can select an isolated database path for tests. Related runtime data is placed beside that database.

Old event rows retain their previous timestamps; new events use UTC. Existing plaintext legacy HDF5 biometric files are not automatically encrypted or migrated by this delivery.

Model/dataset weights and recordings stay out of Git. Dataset/model use terms still apply; generated sample metadata records the primary sources. LLVIP source: https://github.com/bupt-ai-cz/LLVIP (Jia et al., ICCV Workshops 2021).
