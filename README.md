# IntelliGate

IntelliGate is an AI-powered smart gate and security management system for controlled entry points such as homes, offices, campuses, parking areas, and secure facilities. It combines live face recognition, anti-spoofing, vehicle recognition, license-plate OCR, re-identification, and a web dashboard for monitoring and review.

## What It Does

IntelliGate is built around three working layers:

1. Face access control with liveness detection.
2. Vehicle access control with plate recognition and vehicle re-identification.
3. A live dashboard for events, alerts, camera state, and system health.

The backend logs access decisions and suspicious activity so operators can review attempts, track repeat strangers, and inspect historical events.

## Working Features

- Face detection, alignment, and recognition for authorized users.
- Anti-spoofing checks using blink, texture, depth, and mask signals.
- Stranger face logging with repeat-visit detection.
- Vehicle detection, plate detection, and OCR.
- Vehicle re-identification when a plate is unclear or obscured.
- Access decision logging for granted, denied, spoof, and unknown events.
- Live dashboard with stats, recent activity, alerts, and system health.
- WebSocket updates for live event streaming.
- Camera preview and MJPEG streaming through the dashboard backend.

## Where It Can Be Used

### Working use cases

- Residential gate automation.
- Office and company entrance control.
- Campus or university access points.
- Parking lot entry and exit management.
- Restricted zone monitoring in labs, warehouses, or server rooms.
- Security logging for reception desks and guard stations.

### Potential use cases

- Visitor management with pre-registration and audit trails.
- Attendance-style access history for staff or members.
- Multi-camera perimeter monitoring.
- Smart parking with plate-based whitelisting.
- Incident review and evidence collection for security teams.
- Integration with barriers, relays, or turnstiles in a production gate system.

## Project Structure

- `module1_face_recognition/` handles face recognition and anti-spoofing.
- `module2_vehicle/` handles vehicle detection, plate OCR, and re-identification.
- `module8_dashboard/` provides the FastAPI backend and React frontend.
- `config/config.yaml` stores runtime and model settings.
- `logs/` stores runtime event data, uploads, and spoof snapshots locally.

Large local assets such as model weights, generated logs, uploads, build output, and snapshots are intentionally excluded from version control.

## Requirements

- Python 3.10+ recommended.
- Node.js 18+ for the dashboard frontend.
- A webcam or compatible camera for live mode.
- Optional CUDA-capable GPU for faster inference.

## Installation

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Install the frontend dependencies:

```bash
cd module8_dashboard/frontend
npm install
```

## Run the Dashboard Stack

Start the backend API and live event service from the repository root:

```bash
uvicorn module8_dashboard.backend.main:app --reload --port 8000
```

In another terminal, start the frontend:

```bash
cd module8_dashboard/frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5173`.

## Run The Individual Modules

Face recognition module:

```bash
python -m module1_face_recognition.main --mode webcam
```

Register a new face:

```bash
python -m module1_face_recognition.main --mode register
```

Test on a static image:

```bash
python -m module1_face_recognition.main --mode test
```

Vehicle module:

```bash
python -m module2_vehicle.main --mode webcam
```

Register a vehicle:

```bash
python -m module2_vehicle.main --mode register --plate "ABC-1234" --owner "Name"
```

Test vehicle recognition on an image:

```bash
python -m module2_vehicle.main --mode test --image path/to/image.jpg
```

## API and Dashboard Behavior

The dashboard backend exposes REST endpoints for event history, alerts, face records, vehicle records, health monitoring, camera management, and live streams. It also publishes live updates through WebSocket so the UI stays current without page refreshes.

On startup, the backend loads the face pipeline for the default camera stream and can also load the vehicle module and shared databases when the relevant endpoints are used.

## Configuration

Most runtime settings live in `config/config.yaml`:

- camera index, frame size, and device selection.
- face recognition thresholds and database paths.
- anti-spoofing thresholds and model weights.
- dashboard host, port, and CORS settings.

Adjust this file before deployment if you want to change the camera source, processing device, or access thresholds.

## Notes

- Generated files, snapshots, caches, and large assets should stay out of GitHub.
- Keep the code, modules, configuration, and frontend source in the repository.
- If you want a production deployment later, the next step is usually to package the backend, build the frontend, and wire the gate output to a physical controller.