# IntelliGate: software-only delivery plan

Prepared 12 October 2026. Working deadline: 15 October 2026; exact submission time is unspecified.

Subsequent scope change, 12 October: the user explicitly requested live webcam support. The delivered application now allows operator-started webcam inputs in addition to replay, while physical gate/sensor hardware stays excluded. Current operating/testing directions are in `MODULE_TEST_GUIDE.md`. The original plan below is retained as preparation history.

## Agreed constraints and evidence

- Build from the existing `fyp_intelligate` folder.
- Three-day deadline; no GPU and no physical hardware integration.
- Use recorded clips, image sequences and annotated datasets in place of camera hardware.
- Host inspected: Intel i7-1165G7, 4 cores / 8 logical processors, approximately 7.7 GiB RAM.
- Existing source includes face processing, vehicle processing, stranger records, SQLite events, WebSocket updates and a React dashboard. Runtime correctness and accuracy have not yet been verified.
- The HTML audit describes a different repository. Its completion percentages, model scores, service architecture, tests and branch names are not evidence about this checkout.
- `IntelliGate_SRS_v3.pdf` defines ten modules. Home automation appears only as an optional extension in the HTML audit; it is outside this deadline baseline.

## Scope amendment to document in the submission

This is a software prototype using dataset playback and a virtual gate. It does not demonstrate real camera acquisition, thermal sensor operation, physical stopping distances or actuator safety.

Preserve the source SRS and HTML. Record departures explicitly rather than silently rewriting their claims:

| Original requirement | Deadline interpretation | Compliance status |
| --- | --- | --- |
| GPU and IR/thermal/gate hardware | CPU processing, recorded RGB/infrared inputs, virtual device | Hardware excluded by user |
| FR-01: ViT recognition | Benchmark existing ViT; consider a small face-specific pretrained model if baseline fails | Model substitution, if used, must be disclosed |
| FR-06: automatic IR/thermal switching | Low-light profile plus explicit selection of paired recorded modalities | Software routing demonstrated; no sensor switching |
| FR-09: SSM behaviour prediction | Repeat-visit and dwell-time rules | Rule baseline; SSM research deferred |
| FR-12: LLaVA descriptions | Evidence-based event summaries first; lightweight pretrained captioning only if measured feasible | Template reports do not satisfy VLM requirement |
| FR-14: natural-language search | Small documented set of phrases translated into event filters | Limited query parser, not unrestricted language understanding |
| FR-16: PostgreSQL | Retain current SQLite/HDF5 during stabilization | Explicit database deviation; migration deferred |
| BR-01: liveness threshold | Calibrate on validation data and disclose configured threshold | Current config is 0.65, versus SRS reference 0.90; neither establishes accuracy |
| 15 FPS, 1.5-second decisions, 200 ms stop, eight streams | Measure CPU inference and virtual-command latency separately | Targets remain unverified until benchmarked |
| 99%/99.4% OCR, 97% Re-ID, 99.5% uptime | Report measured metrics with sample size and limitations | No guarantee or invented compliance |

The user's engineering scope change does not establish academic acceptance of these SRS departures. Include this amendment with the submission for supervisor review.

## Module coverage and acceptance matrix

All ten modules remain visible in the plan. These are proposed deadline deliverables, not claims of completed implementation.

| Module / requirements | Software deliverable | Acceptance evidence | Limits |
| --- | --- | --- | --- |
| M1 / FR-01..03 | Enrolment, face match, liveness checks, denied-spoof events, approval on uncertainty | Held-out genuine/spoof samples; known and unknown clip demonstrations | Static images cannot validate blink/replay resistance; image-derived depth is not measured sensor depth |
| M2 / FR-04..05 | Vehicle registry, plate OCR, appearance candidate matching when OCR fails | Plate exact-match and character error rates; separate Re-ID query/gallery results | Generic ImageNet embeddings are a baseline; appearance alone does not authorize opening |
| M3 / FR-06 | Low-light enhancement comparison and recorded infrared/thermal evaluation | Fixed annotated subset; baseline versus enhancement detection results | RGB colour maps are not thermal data; pretrained RGB detectors must be evaluated for modality shift |
| M4 / FR-07..09 | Stranger memory, distinct visits, configurable repeat alerts and dwell rules | Three separate visits trigger one escalation; continuous frames do not create new visits | SSM prediction deferred; heuristic alerts do not prove intent |
| M5 / FR-10 | ROI queue counts, density thresholds and congestion events | Compare unique tracked counts with annotations on short sequences | No trained congestion predictor; report tracking errors |
| M6 / FR-11 | Gate-path ROI obstruction and virtual STOP/lockdown | Obstruction blocks OPEN/CLOSE; STOP overrides pending motion; event timeline records result | Software interlocks only, no physical safety claim |
| M7 / FR-12,14 | Daily evidence-based reports, event summaries and limited query parser | Every statement links to stored events; unsupported query returns clear guidance | Label template output; VLM/SSM components deferred unless genuinely integrated and tested |
| M8 / FR-13 | Playback preview, logs, alerts, registry management, gate panel, module status | Browser walkthrough and frontend build; updates without refresh | Responsive web interface; no separate native mobile app baseline |
| M9 / hardware replacement | Virtual gate with OPEN, CLOSED, OPENING, CLOSING, STOPPED and LOCKDOWN states | Deterministic transition tests, expiry/idempotency, obstruction interlock | Simulation labels on every device screen and command result |
| M10 / FR-15,16 and security NFRs | Existing FastAPI backend, durable event storage, configuration, health and local reproducible launch | API tests, restart persistence, validation, auth checks, CPU-only launch | SQLite deviation disclosed; encryption, RBAC, TLS and immutable audit claims require implementation evidence |

## Architecture for this deadline

Retain FastAPI, React and existing storage while establishing common interfaces. Avoid a new Valkey/PostgreSQL/SeaweedFS service stack during this deadline.

Flow: uploaded clip or dataset sequence -> frame sampler -> detector/tracker -> module observations -> central access/safety policy -> virtual gate -> persistent events -> WebSocket/dashboard -> report.

- Process one heavy inference job at a time initially. Bound queues; lazy-load models; use smaller input sizes only after measuring accuracy impact.
- Separate playback FPS from inference FPS. Label frame skipping and prerecorded inputs.
- Define one event envelope: event ID, UTC timestamp, source ID, module, event type, track/session ID, confidence, decision, reason, evidence reference and simulation flag.
- Store video source time separately from processing time. Benchmark end-to-end delay independently from inference duration.
- Use dataset/clip source timestamps for replay visit windows; never merge simultaneous cameras solely by local track ID.
- Centralize decisions: ALLOW, DENY and REVIEW_REQUIRED. Obstruction/lockdown override ordinary access. Uncertain identity and failed checks require review.
- Enrolment and database changes require administrator authorization. Apply the same authorization to REST and WebSocket actions.
- Model replacements require embedding versioning and re-enrolment/re-indexing when dimensions or preprocessing change.
- Preserve source provenance and avoid sending raw biometric embeddings to browser clients.

## Dataset shortlist and access gates

Web sources checked 12 October 2026. These are candidates, not downloaded or verified local assets. Start with a small fixed subset, record IDs/hashes and terms, and preserve official splits. Do not tune on the evaluation subset.

| Task | Candidate and primary source | Initial experiment | Access / interpretation |
| --- | --- | --- | --- |
| Paired low-light/infrared | [LLVIP](https://github.com/bupt-ai-cz/LLVIP) | 100-200 held-out annotated pairs; visible baseline/enhanced visible/infrared results | Check linked use terms and download access; select weights appropriate to the modality |
| Thermal pedestrians/vehicles | [FLIR ADAS](https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/) | 100-200 annotated frames; detection precision/recall and latency | Download form and terms can introduce delay; thermal intensity is not automatically calibrated temperature |
| Tracking/queue | [MOT17 archive](https://motchallenge.net/data/MOT17/) | One short training sequence with fixed ROI and manual count checks | Archive lists a 5.5 GB download; timebox retrieval; pedestrians only, not a vehicle queue benchmark |
| Plate OCR | [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset) | 100-200 held-out annotated plates; exact match and character error rate | Check current access terms; Brazilian plates do not establish Pakistani plate accuracy |
| Image anti-spoofing | [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | Small genuine/attack subset; APCER/BPCER and confusion matrix | Verify terms, weight preprocessing and subject split; still images do not evaluate blink liveness |
| Local video playback | `D:/fyp/nre/video test.mp4` and `video test 2.mp4` | Inspect content and establish permitted use before choosing demo scenarios | Existing files are not automatically labelled or consented benchmark data |

[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) is not the default: its current terms require written confirmation for projects sponsored by a for-profit entity. The SRS names Falcon Systems as sponsor. Underlying image rights also differ from software licensing.

If dataset access is unavailable within two hours, proceed with permitted local clips for functional demonstrations and mark quantitative dataset evaluation pending. Do not relabel those clips as a published benchmark. Thermal benchmark remains unavailable until genuine modality data is obtained.

No deep-model fine-tuning is on the critical path. CPU threshold calibration and lightweight classifiers on cached embeddings are possible only after an initial timed experiment; they must use a separate validation split and held-out evaluation. Never label calibration as fine-tuning a deep network.

## Three-day order of work

### Day 1: 12 October - establish a working core

1. Verify Python/Node environments and dependency compatibility; run current frontend/backend and capture failures.
2. Inspect local clips and model availability. Timebox dataset acquisition and baseline CPU benchmarks.
3. Stabilize video playback, model startup and event persistence.
4. Implement central policy and virtual gate with safety interlocks.
5. Demonstrate one genuine known/unknown face flow and one vehicle flow with persisted events. Keep failing recognition checks visible.

### Day 2: 13 October - integrate missing software modules

1. Add stranger visit semantics, ROI queue counts and obstruction events.
2. Add low-light processing and genuine infrared/thermal inputs if access succeeded.
3. Add reports, limited search, virtual gate controls and dashboard module status.
4. Complete enrolment/registry workflows, authentication and role restrictions; exercise negative paths.
5. Verify each module against the acceptance matrix. Mark any deferred model-specific requirement explicitly.

### Day 3: 14 October - freeze and prepare submission

1. Run held-out subset benchmarks and record model versions, sample IDs, hardware and timing scope.
2. Test restart persistence, repeated commands, invalid inputs and obstruction override.
3. Fix critical demo blockers; build frontend and verify a clean CPU launch.
4. Record the full demonstration; prepare evaluation tables, limitations and SRS amendment.
5. Freeze features and leave 15 October for submission and emergency fixes.

## Submission checklist

- Reproducible CPU launch instructions and dependency lock/constraints.
- A scenario manifest listing each clip, provenance, expected events, annotations and module coverage.
- Screens show simulation, replay, baseline and pending status accurately.
- Benchmark report with subset sizes, separate validation/evaluation, errors, FPS and latency.
- No hard-coded success scores, fabricated research results or fake thermal inputs.
- Saved demonstration video, screenshots, known limitations and explicit deferred requirements.
- All ten modules accounted for; software coverage and original-SRS compliance reported separately.

## Next implementation task

Run the existing stack and benchmark one face and one vehicle input on CPU. Use those measurements to select practical inference models and frame sampling before adding new module code.
