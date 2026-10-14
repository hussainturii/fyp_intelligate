# Test Center: diagnose the live feed and test with supplied data

Updated 14 October 2026.

## Fixed camera-startup failure

A cold webcam can take longer than 0.5 seconds to deliver its first frame. The previous recognition worker treated a condition-variable timeout as a ready frame and called `.copy()` on `None`. That killed recognition while the separate capture thread kept displaying live video. A failure-first regression test reproduced the exact crash; it passes after the worker was changed to wait until a frame actually exists.

Camera / Replay now shows the active mode, worker status, number of processed frames and detected/known/unknown faces. A preview picture by itself is no longer presented as proof that recognition is running.

The existing registered profiles are preserved. Do not delete/re-enrol them merely to recover from this startup problem.

Post-fix verification on 14 October: a six-second real-webcam check using an isolated copy of the two saved profiles processed 27 frames, matched one registered face, emitted one appearance event and reused the identity cache 26 times. Capture measured approximately 20 FPS and detector throughput 4.8 FPS in that short check. Both production profiles were preserved and no camera images were retained by that check. The software test suite passed 45 tests. The component runner reported 10 narrow checks PASS, 4 research features NOT_IMPLEMENTED and 4 evaluations NEEDS_DATA; this is not a claim of full project completion or measured identification accuracy.

## Start or restart the corrected version

Double-click `D:\fyp\fyp_intelligate\start_intelligate.bat`. This runs the server in a visible terminal; Ctrl+C in that terminal stops it. If another backend is already running, stop that instance before opening the launcher again. Refresh http://127.0.0.1:8000 after startup so the new dashboard bundle loads.

Alternatively, in PowerShell:

```powershell
cd D:\fyp\fyp_intelligate
python -m uvicorn module8_dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

## Diagnose your own live face

1. Sign in and open Camera / Replay in face mode. Start camera index 0.
2. Confirm **worker: running** and **processed frames** increasing. Raw mode deliberately does not recognize faces.
3. If faces detected is zero, the worker is alive but the detector does not see a usable face. Improve lighting, camera angle or distance. If the worker is stopped/error/stalled, inspect the displayed error instead of changing enrolment photos blindly.
4. Open **Test Center -> Diagnose the current camera frame**. Enter source index 0 and optionally the exact registered name you expect. Click **Test current camera frame**.
5. Inspect the annotated image, detected faces, matched labels and similarity scores. The threshold currently used is 0.45 and is provisional. A low score should be investigated with separate photos, not fixed by blindly lowering the threshold.
6. This snapshot test runs matching independently of the live worker, against an isolated copy of the registered gallery. It therefore distinguishes a detection/matching issue from a capture-thread issue. The report shows how many gallery entries were copied.

If the same person has been registered under two different names, the best-scoring name may vary between those aliases. Expected-name assertions compare the exact profile label; they do not infer that two names refer to one person.

## Check the project components

Click **Run component checks** on Test Center. Each row names the actual scope of its check:

| Module | Functional check | Remaining evidence/research |
| --- | --- | --- |
| M1 | Real detector + temporary encrypted enrolment + self-match fixture | Held-out face accuracy needs supplied data; validated anti-spoofing is not implemented |
| M2 | Temporary vehicle registration and exact/unknown plate lookup | OCR accuracy needs readable labelled clips; trained Re-ID is not implemented |
| M3 | Finite low-light enhancement output on a dark fixture | Genuine infrared detection accuracy needs annotated data |
| M4 | Distinct-visit counting and repeat alert | Real identity association needs data; SSM prediction is not implemented |
| M5 | Two stable synthetic tracks and counts | Real queue-count accuracy needs labelled clips/ROIs |
| M6 | Obstruction stops/blocks virtual motion | No physical actuator safety validation |
| M7 | Generated report links to stored evidence | VLM captions are not implemented |
| M8 | Production frontend assets exist | This is not a browser interaction test |
| M9 | Virtual OPEN transition and STOP precedence | Physical hardware is excluded |
| M10 | Password/session checks, logout and event persistence | HTTP role restrictions are tested by the automated pytest suite |

**PASS** means the named narrow check passed. **FAIL** means it failed. **NOT_IMPLEMENTED** identifies missing functionality; **NEEDS_DATA** identifies an evaluation which the built-in fixture cannot establish. A page existing or a self-match working does not mean that all original SRS requirements are met.

## Supply an image or video

On Test Center -> Test with your own data:

1. Choose Face, Vehicle, Scene, Low light or Infrared.
2. Upload a supported image or video. Use a different photo/video from the enrolment image for a meaningful face test.
3. For face tests, optionally supply the expected registered name or `Unknown`. An expected face count can also be supplied.
4. For scene/vehicle/infrared tests, optionally supply an expected object count. Counts apply to every sampled frame, so use a short stable segment or leave the expectation blank for varying scenes. These counts are ROI counts, not necessarily every object in the entire original frame.
5. Set maximum sampled frames (1-300; default 30). Video sampling starts at the beginning at approximately five samples per source second. For the default 30-second dwell rule, use a clip longer than 30 seconds and at least 180 samples. Source timestamps drive tracking/dwell calculations; processing does not pretend the whole clip happened at one wall-clock instant.
6. Wait for the run to finish. Inspect the annotated last frame, all sampled face labels/scores or object counts, obstruction flags, number/types of events and measured latency.
7. Download the complete JSON evidence. An expectation that fails produces **FAIL** even when inference executed without errors. With no expectation, the verdict is **OBSERVED_NO_EXPECTATIONS**, never an automatic accuracy pass.

Tests copy the current gallery into temporary storage. Test strangers, approvals, visits and gate movements stay in temporary state; they do not alter the live gallery, timeline or virtual gate. Uploaded input files are removed after completion. Annotated previews and JSON reports remain locally under `logs/testing/<run-id>/` and are accessible to signed-in users in this local application. Avoid including people who have not agreed to your testing.

Stop live sources before timing benchmarks so competing inference does not distort measurements. Only one workbench job runs at a time. If the backend is stopped during a job, its report is marked interrupted on restart and the test must be run again.

## Useful test data and expected observations

- **Face:** separate frontal photos of enrolled and unenrolled people; then a clip with three/four clear faces. Check correct labels, false matches, misses and event counts, not only whether a box is drawn.
- **No-flooding:** use a 10-20 second continuous clip. A stable identity should produce one appearance event, with recognition cache hits increasing. A true gap longer than two seconds should permit a new appearance event.
- **Vehicle:** readable plates with ground-truth text, unknown vehicles and clips where plates become unreadable. Current appearance matching is a colour-histogram baseline; contour plate localization can fail. Inspect OCR/method metadata in JSON. Do not claim 99% OCR without a labelled evaluation.
- **Low light:** compare the same scene in scene/raw/low-light workflows. CLAHE changing pixels is not proof that detection improved.
- **Infrared:** genuine infrared imagery only. Prepared LLVIP illustrations are useful for visual demonstrations but lack ground-truth labels for accuracy claims.
- **Safety/queue:** clips with a known count inside a documented ROI and objects crossing the virtual gate path. Interactive ROI controls and current-source tests are described in `MODULE_TEST_GUIDE.md`; upload evaluation currently uses the documented default ROIs.
- **Spoofing:** a photograph may be recognized as a person's identity. The current CPU path requires review and does not establish anti-spoof detection. The test matrix deliberately marks validated liveness unavailable.

## Command-line equivalents

```powershell
python tools/run_project_checks.py
python -m pytest tests -q
```

The first writes `logs/testing/component_checks_latest.json` and explicitly lists missing research/data checks. The second exercises software regressions and API/role behaviour; it does not establish real-world model accuracy. For the longer manual module walkthrough, see `MODULE_TEST_GUIDE.md`.
