# IntelliGate live-camera and module test guide

Updated 12 October 2026 after the user requested live webcam support. CPU remains the inference device and the gate remains virtual. Webcam capture is now explicitly supported; it starts only when the operator presses Open live camera.

## Open the application

Open http://127.0.0.1:8000 and sign in. If the backend is not running:

```powershell
cd D:\fyp\fyp_intelligate
python -m uvicorn module8_dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

If frontend source changed, build it first with `npm run build` inside `module8_dashboard/frontend`. The backend serves the built dashboard. Use a single hostname consistently for session cookies.

Only one live/replay source runs at a time. Opening a new source stops the previous source. Viewer accounts can inspect data but cannot start/control sources, enrol members or open the gate.

## Live face recognition: use this first

1. On **Camera / Replay**, select **Face candidates**, leave webcam index **0**, and press **Open live camera**.
2. Confirm a LIVE preview appears. Index 0 usually means the built-in camera; try index 1 if an external USB camera is attached.
3. For camera enrolment, keep only one person visible, look toward the camera in good light, then go to **Face Database -> Enrol from the current camera frame**. Enter that person's name and use source index 0. Repeat separately for each participant. Alternatively upload one clear frontal image per person.
4. Return to Camera / Replay. Registered people should have **green boxes with names**; clear unregistered faces should have **red boxes labelled Unknown**. Very small face crops display an amber move-closer message instead of storing an unreliable match.
5. Stand three or four people in the frame. Each detected face has its own track number and independent label. Do not overlap faces; ensure faces are large enough and reasonably frontal.
6. Note the **identity/OCR runs** counter. Once each face has a result, that counter should remain stable while **cache hits** increase. The detector still refreshes boxes independently.
7. Open **Event Log** and select module face. Expect one recognition_candidate or stranger_alert per distinct continuous identity appearance, not an event every few frames. Multiple simultaneous events are merged into the live list.
8. Remain visible for 30-60 seconds. No repeated face recognition events should appear just because time passes.
9. Step out of view for **more than two seconds**, then return. A new track, recognition run and appearance event are expected. A brief missed detection or occlusion shorter than the grace period normally keeps the cached result.
10. After testing, press **Stop source**. The camera is not started automatically on backend launch.

### Recognition frequency and event semantics

- Capture and JPEG preview run independently of the heavy inference worker. The worker keeps only the newest pending frame; it does not accumulate a delayed camera backlog.
- Face/object detection is capped at **five passes per second**. Actual inference FPS is displayed and depends on CPU load. Preview FPS is a separate number.
- Face embedding/gallery matching runs once per usable stable track, then its name/result is cached.
- A cached **known** identity is not periodically re-recognized. Track loss, ambiguous overlap/crossing, an abrupt crop-appearance change, or a gallery change can require a fresh identity check. A cheap image fingerprint helps detect abrupt replacement of a face at the same position; it is not a guarantee against identity switches. Rechecking a continuously present identity does not itself repeat its event.
- Unknown faces/vehicles can receive up to three recognition attempts only when their crop becomes materially larger (at least 1.5x), so an early unreadable distant input does not remain permanently unknown. An unknown -> known change is a meaningful new result.
- An appearance ends after two seconds without a usable associated detection. This is based on tracking visibility, not proof that a person physically left the room. Turning away for a long time can therefore create a new appearance on return.
- Recognition events and access approval are separate. **A green face box means a gallery match, not validated liveness or permission to open the gate.** Face access still requires operator review.
- Every source restart creates a new tracking session. Initial events after a deliberate restart are expected.

### What has been verified

The local webcam opened at 960x540 and a real captured frame produced one face detection (approximately 128 ms detector time in that check). Automated tests exercise four independent face tracks, cached recognition, colours and event suppression. This is not a measured four-distinct-person recognition-accuracy study; complete the multi-person manual test above with your enrolled participants.

In a separate six-second live-stream check, preview capture measured approximately 19.5 FPS and detector processing 4.6 FPS. One continuously visible face produced one identity run, 25 cache hits and one face event. These are short-run CPU measurements, not an accuracy or long-term reliability claim. The final automated suite passed 37 tests; the production frontend build passed.

## Vehicle recognition and event suppression

1. Register a plate and owner on **Vehicle DB**. Use the real plate text, normalized to letters/digits. Optionally upload a **cropped image of that vehicle**, rather than an entire street scene.
2. On Camera / Replay select **Vehicle detection / OCR**. Use a relevant recorded traffic/gate clip, or open a live camera that actually sees vehicles.
3. Each detected car/truck/bus/motorcycle is tracked separately. A sufficiently confident exact registered plate produces a green owner/plate label. Unknown vehicles are red; appearance-only candidates are amber and require review.
4. Keep the same vehicle visible. The cached result should remain labelled, and its OCR/identity counter should stop increasing once classified. Its event should not repeat every ten seconds.
5. Remove the vehicle for more than two seconds and bring it back: a new track and event are expected.
6. To test plate fallback, use a clip with an unreadable plate and an optional registered crop. Any appearance candidate must request review; it must not automatically open the virtual gate.
7. In Event Log inspect event metadata through the downloaded evidence report: plate, OCR confidence, method, source and track ID explain the result.

**Current limits:** plate localization uses an OpenCV contour fallback because the old specialized model URL was unavailable. EasyOCR accuracy has not been benchmarked on your plate format. Colour histograms are a weak appearance baseline, not trained vehicle Re-ID. A miniature toy car or a car on a laptop screen is not a valid real-world plate-accuracy evaluation. Use readable, labelled clips and count false positives/misses honestly.

## Test all ten modules

| Module | Steps in the dashboard | Expected working result | What this does not establish |
| --- | --- | --- | --- |
| M1 Face recognition / liveness | Run the live known/unknown and 3-4-person tests above. Show a photograph of a registered person as a separate negative test. | Names/colours and one event per appearance; photo input must not automatically open the gate because face access requires review. | Photograph recognition can still occur. This does not demonstrate a trained anti-spoof detector rejecting attacks. |
| M2 Vehicles | Register a real plate, run Vehicle mode on a relevant clip, then hide/unread the plate. | Plate/owner candidate or red unknown box, cached results and no repeated entry events; appearance fallback requests review. | 99% OCR or 97% Re-ID accuracy is not established. |
| M3 Low light / infrared | Replay the LLVIP visible illustration in raw mode, then low-light enhancement. Replay the genuine LLVIP infrared illustration in thermal mode. | RGB CLAHE visibly changes low-light input; real infrared imagery is decoded and detection/count status is shown. Low-luminance scene input emits a profile event. | RGB false-colour is not thermal data. This illustration is unannotated; no thermal accuracy or calibrated temperature claim. |
| M4 Stranger memory | A clear unenrolled participant enters, leaves for >2 seconds, and returns three times within 24 hours. Inspect Operations -> Stranger visits and Alerts. | Stable unknown candidate ID where matching succeeds; visits increase per appearance; third distinct visit escalates once. Continuous standing should not add visits. | Unknown matching is provisional; no SSM or threat-intent prediction. |
| M5 Queue / density | Select scene mode on a pedestrian/vehicle clip. In Operations draw the normalized queue ROI around the desired area and set threshold 2 or 3. | Blue ROI, displayed count, count-change history and congestion event/alert when threshold is reached. | Counts depend on detector/tracker quality; no trained congestion forecasting. |
| M6 Safety | In Operations simulate obstruction, try OPEN and CLOSE, then clear obstruction and RESET. Repeat with scene mode and a visible person/vehicle intersecting the red gate ROI. Press STOP during OPENING/CLOSING; test LOCKDOWN. | Obstruction stops/blocks virtual motion; STOP cancels pending motion; LOCKDOWN blocks opening; events/alerts persist. | Software interlocks are not certified physical safety or proof of sensor-to-actuator 200 ms performance. |
| M7 Reports / search | After generating real events, click Generate last 24 hours and download evidence JSON. Search `face today`, `vehicle`, `queue`, `obstruction`. Try an unsupported sentence. | Report counts match stored events; evidence contains their IDs; supported queries return matching rows; unsupported query explains limitations. | Reports use templates, not VLM captions; search is a keyword parser. |
| M8 Dashboard | Open two signed-in browser tabs, generate an appearance or obstruction, then inspect logs/alerts in the other tab. Refresh the page. | Live event updates, persisted history after refresh, available enrolment and source controls. | End-to-end push latency must be measured if quoted in the report. |
| M9 Virtual device | OPEN -> OPENING -> OPEN -> CLOSING -> CLOSED; STOP in motion; LOCKDOWN then RESET after clearing obstruction. | State transitions and command audit events. Repeating a command ID through the API does not execute twice. | No physical relay, motor, temperature sensor or gate is connected. |
| M10 Backend / security | Create viewer/security_staff accounts from Operations. Sign in as viewer and try a write; sign in as staff and try enrolment. Restart backend after recording events. | Viewer writes and staff enrolment are denied; stored events/galleries survive restart; active gate motion becomes STOPPED on restart. | No production TLS, PostgreSQL migration, immutable audit anchoring or long-term availability proof. |

## Record pass/fail evidence

For each test record source/participants, enrolment images, date, expected result, observed result, errors and a screenshot/video. For recognition record correct names, wrong names, misses and unknowns separately. Do not count a matched photo as a liveness pass. Keep demo data and screenshots local unless participants have agreed to sharing.

The **Operations -> Module implementation status** panel lists active implementations and limitations. A module page being present is not evidence that every original SRS research requirement has been met.

If the scene pipeline still sees an object in the red gate ROI, clearing the operator's simulated obstruction will not clear that camera obstruction. Move the object outside the ROI or stop the source, then clear/reset the virtual gate.

## Automated verification and benchmarks

```powershell
python -m pytest tests -q
python tools/benchmark_cpu.py --video "D:\fyp\nre\video test.mp4" --frames 8
python tools/benchmark_llvip.py
```

Optional real-model tests require `python tools/download_face_models.py` and `python tools/prepare_demo_data.py`. Benchmarks write JSON to `logs/benchmarks`. Timings are not recognition-accuracy measurements.

## Troubleshooting

- **Camera will not open:** close Teams/Zoom/Windows Camera; check Windows Settings -> Privacy & security -> Camera -> allow desktop apps; try index 1; Stop source before reopening.
- **No boxes:** choose face mode, use a reasonably frontal view and good light, and move closer. In raw mode no recognition runs. Low-light and thermal modes are different processing paths.
- **Wrong person name:** stop using that match as an access signal; improve the enrolment image and evaluate the provisional threshold. Record false matches. Green boxes do not bypass the review requirement.
- **Events appear again:** check whether the face was lost for >2 seconds, source restarted, registry changed, or tracking became ambiguous. Persistent stable tracks should not emit timed repeat notifications.
- **Live preview remains fast but detector FPS is lower:** this is expected with separate capture/inference. Check CPU usage and displayed detector latency. The full-pipeline 15 FPS target has not been demonstrated.
- **Infrared demo missing:** run `python tools/prepare_demo_data.py`. Those clips are the official LLVIP illustration, not the complete labelled dataset.
