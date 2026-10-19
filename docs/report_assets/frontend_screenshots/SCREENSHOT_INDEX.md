# IntelliGate Frontend Operational Interface — Final FYP Report Screenshot Index

**Project Title:** IntelliGate: Intelligent Multimodal Access Control & Security Surveillance System  
**Institution:** COMSATS University Islamabad, Department of Computer Science (BS Artificial Intelligence)  
**Deliverable:** Academic Report Assets — High-Resolution Frontend Production Screenshots  
**Asset Directory:** `docs/report_assets/frontend_screenshots/`  
**Display Format:** 1920 × 1080 Full-HD Uncompressed PNG (`scale: 1.0`, dark operations theme)  
**Provenance:** Captured directly from the running IntelliGate FastAPI backend (`main:app`) and React 18 production bundle (`module8_dashboard/frontend/dist`) via headless Chromium automation.

---

## 1. Executive Summary & Quality Assurance

This document indexes the 15 production screenshots of the IntelliGate web application captured for inclusion in the final Bachelor of Science in Artificial Intelligence Final Year Project (FYP) report. 

In strict adherence to academic rigor, scientific integrity, and privacy ethics:
1. **Zero Data Fabrication:** All displayed metrics, access logs, face enrolment entries, and vehicle registrations originate from actual operational database tables (`logs/events.db`, `logs/cpu_vehicles.h5`, and AES-256 GCM encrypted SFace feature store).
2. **State Machine Verification:** All four primary gate state machine branches (`ALLOW` / `OPENING`, `REVIEW_REQUIRED`, `STOPPED` with physical safety obstruction, and `LOCKDOWN`) are accurately visualized under live software runtime execution.
3. **Privacy & Security Preservation:** No cleartext passwords, cryptographic keys, personal biometric embeddings, or private institutional credentials are leaked in any UI component. Biometric feature vectors remain encrypted at rest via AES-256 GCM.
4. **Visual Consistency:** Every screenshot was rendered under a standardized $1920 \times 1080$ desktop viewport with uniform typography, consistent dark glassmorphic styling, and fully legible SVG chart visualizations.

---

## 2. Master Screenshot Inventory

| # | Filename | Screen / Component | Operating State Demonstrated | FYP Report Section | Status |
|:---:|:---|:---|:---|:---|:---:|
| **01** | `01_login.png` | Authentication Gateway | Security Pre-Auth Barrier | Chapter 3: System Design & Auth | Verified |
| **02** | `02_main_dashboard.png` | Central Overview Dashboard | Normal Monitoring Telemetry | Chapter 4: Central Operations Console | Verified |
| **03** | `03_face_recognition.png` | Biometric Face Enrolment & Registry | Enrolled Identity Management | Chapter 4: Facial Biometric Pipeline | Verified |
| **04** | `04_vehicle_recognition.png` | Vehicle DB & Plate Whitelist | Registered Fleet Registry | Chapter 4: Automated ALPR Pipeline | Verified |
| **05** | `05_virtual_gate_controls.png` | Virtual Gate Operator Console | Hardware Simulation Ready | Chapter 4: Software Gate Actuation | Verified |
| **06** | `06_camera_replay_monitoring.png`| Multi-Stream Vision / Replay | Live Video Feed & Offline Ingestion | Chapter 4: Vision Telemetry & Replay | Verified |
| **07** | `07_alerts_security_stranger.png` | Security Alert Operations Center | Multi-Severity Threat Incident Feed | Chapter 4: Threat Detection & Alerts | Verified |
| **08** | `08_queue_scene_monitoring.png` | Scene Rules, Queue ROI & Tracking | Congestion ROI & Stranger Dwell | Chapter 4: Spatial Scene Analytics | Verified |
| **09** | `09_event_history_audit.png` | Access Event Log Audit Trail | Multimodal Chronological Audit | Chapter 4: Security Audit & Storage | Verified |
| **10** | `10_reporting_evidence_analytics.png` | Evidence Reporting & Event Search | 24h Summary & Natural Search | Chapter 4: Evidence & Analytics | Verified |
| **11** | `11_gate_state_allow_opening.png` | Gate State Machine: ALLOW | `OPENING` State Transition | Chapter 5: Operational State Machine | Verified |
| **12** | `12_gate_state_review_required.png` | Gate State Machine: REVIEW_REQUIRED | Operator Discretionary Review | Chapter 5: Human-in-the-Loop Review | Verified |
| **13** | `13_gate_state_stopped_obstruction.png` | Gate State Machine: STOPPED | Safety Interlock Obstruction Arrest | Chapter 5: Fail-Safe Safety Policies | Verified |
| **14** | `14_gate_state_lockdown.png` | Gate State Machine: LOCKDOWN | High-Security Facility Lockdown | Chapter 5: Emergency Protocols | Verified |
| **15** | `15_test_center_diagnostics.png` | Diagnostic Test Center | Modular Component Health Check | Chapter 6: Verification & Testing | Verified |

---

## 3. Detailed Screenshot Specifications & Captions

### Screenshot 01: Login Gateway
- **Relative Path:** `docs/report_assets/frontend_screenshots/01_login.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 3.4: IntelliGate Secure Authentication Gateway featuring role-based access control (RBAC), multi-role credentials (Security Admin, Station Guard, Compliance Auditor), and automated session token issuing.*
- **Technical Description:** Displays the centralized login interface enforcing authenticated HTTP-only session tokens before any telemetry or gate controls become accessible.

---

### Screenshot 02: Main Overview Dashboard
- **Relative Path:** `docs/report_assets/frontend_screenshots/02_main_dashboard.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.1: IntelliGate Central Security Operations Dashboard presenting real-time system health telemetry (CPU utilization, memory allocation, replay FPS sparkline), biometric/vehicle registry counts, open security alerts, and live event feed.*
- **Technical Description:** Highlights the primary operator console. Shows 4 enrolled facial identities, 5 registered vehicles, 6 active security alerts, CPU/Memory telemetry sparklines, and the 10 most recent verified access transactions.

---

### Screenshot 03: Face Recognition & Enrolment Management
- **Relative Path:** `docs/report_assets/frontend_screenshots/03_face_recognition.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.5: Biometric Management Console illustrating single and batch facial enrolment, dynamic identity lookup, and database summary of registered personnel with AES-256 GCM encrypted feature vectors.*
- **Technical Description:** Shows the administrative interface for enrolling authorized individuals. Emphasizes that facial embeddings are computed via SFace (128-dimensional vectors) and encrypted at rest with AES-256 GCM authenticated encryption.

---

### Screenshot 04: Vehicle Recognition & License Plate Registry
- **Relative Path:** `docs/report_assets/frontend_screenshots/04_vehicle_recognition.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.8: Automated License Plate Recognition (ALPR) Management Interface depicting authorized vehicle registrations, plate whitelisting (Pakistani provincial alphanumeric formats), vehicle ownership classification, and registration metadata.*
- **Technical Description:** Demonstrates the HDF5/SQLite backed vehicle registry containing authorized Pakistani format plates (`LEA-1234`, `ICT-5678`, `MNL-8185`), allowing fast $O(1)$ hash lookups for barrier actuation.

---

### Screenshot 05: Virtual Gate Controls & Operations
- **Relative Path:** `docs/report_assets/frontend_screenshots/05_virtual_gate_controls.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.11: Software Virtual Gate Operations Panel providing manual override actuation controls (`OPEN`, `CLOSE`, `STOP`, `LOCKDOWN`, `RESET`), simulated optical beam obstruction triggers, and human-in-the-loop discretionary access review.*
- **Technical Description:** Renders the central operations panel for virtual barrier control. Includes operator controls, the active status display (`CLOSED`), safety sensor obstruction simulation, and pending access reviews.

---

### Screenshot 06: Camera & Replay Stream Monitoring
- **Relative Path:** `docs/report_assets/frontend_screenshots/06_camera_replay_monitoring.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.3: Real-Time Video Stream and Offline Replay Ingestion Console demonstrating camera source selection, live feed display, offline video dataset processing controls, and frame rate benchmarking indicators.*
- **Technical Description:** Shows the computer vision ingestion console capable of multiplexing between live RTSP/USB hardware camera devices and pre-recorded evaluation datasets with adjustable replay speeds.

---

### Screenshot 07: Security Alerts & Stranger Anomaly Detection
- **Relative Path:** `docs/report_assets/frontend_screenshots/07_alerts_security_stranger.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.14: Security Alerts and Incident Management Center categorizing threats by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), including hardware interlock trips, biometric presentation attacks (spoof detection), unauthorized vehicle entry attempts, and recurrent visitor tracking.*
- **Technical Description:** Demonstrates the automated incident logging engine. Illustrates clear severity badges, detailed root-cause descriptions, timestamps, and manual resolution workflows for security officers.

---

### Screenshot 08: Queue, Scene Rules & Visitor Tracking
- **Relative Path:** `docs/report_assets/frontend_screenshots/08_queue_scene_monitoring.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.17: Spatial Scene Analytics Panel displaying Region of Interest (ROI) configuration for traffic queue monitoring, virtual gate perimeter tripwires, dwell time loitering thresholds, and recurrent stranger visit clustering.*
- **Technical Description:** Illustrates the spatial analytics engine where operators configure normalized bounding coordinates $[0, 1]$ for vehicle queue counting and safety curtains, alongside continuous stranger appearance tracking.

---

### Screenshot 09: Event History & Multimodal Audit Trail
- **Relative Path:** `docs/report_assets/frontend_screenshots/09_event_history_audit.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.19: Comprehensive System Audit Trail recording timestamped access decisions, detection modalities (face recognition, plate OCR, hardware interlock), identity matches, and quantitative confidence score bars.*
- **Technical Description:** Displays the chronological database audit log in `events.db`. Each row features a visual confidence score bar ($0\% - 100\%$), camera source index, verification method, and associated metadata.

---

### Screenshot 10: Evidence Reporting & Natural Language Query Search
- **Relative Path:** `docs/report_assets/frontend_screenshots/10_reporting_evidence_analytics.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 4.22: Forensic Evidence Reporting and Natural Language Event Search Interface enabling rapid 24-hour incident report generation, access summary breakdown, and structured audit query filtering.*
- **Technical Description:** Highlights the operational reporting interface. Operators can click a single button to generate structured 24-hour audit summaries or execute parametric queries (`authorized vehicle`, `face today`) against historical event logs.

---

### Screenshot 11: Gate Operating State — ALLOW (`OPENING`)
- **Relative Path:** `docs/report_assets/frontend_screenshots/11_gate_state_allow_opening.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 5.2: Gate Finite State Machine in `ALLOW` transition: barrier status displays dynamic `OPENING` state with safety obstruction sensors confirmed clear (`Obstruction: clear · Lockdown: off`).*
- **Technical Description:** Validates the positive access pathway. Triggered by a verified biometric or license plate match, the state machine transitions from `CLOSED` to `OPENING` with safety clearance.

---

### Screenshot 12: Gate Operating State — REVIEW_REQUIRED (Human-in-the-Loop)
- **Relative Path:** `docs/report_assets/frontend_screenshots/12_gate_state_review_required.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 5.4: Human-in-the-Loop Discretionary Review Interface: displaying an active `REVIEW_REQUIRED` state with an operator decision card (`Dr. Sarah Qureshi · event #14`) featuring one-click `Approve` and `Deny` buttons.*
- **Technical Description:** Proves the dual-threshold security policy. When facial similarity or ALPR confidence falls within the discretionary range $[0.50, 0.65)$, the gate remains secure while alerting human guards for manual confirmation.

---

### Screenshot 13: Gate Operating State — STOPPED (Safety Obstruction Active)
- **Relative Path:** `docs/report_assets/frontend_screenshots/13_gate_state_stopped_obstruction.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 5.6: Safety Interlock Actuation: gate state machine in `STOPPED` condition following optical safety curtain beam interruption (`Obstruction: present`), enforcing immediate motion arrest.*
- **Technical Description:** Validates the physical safety interlock requirement. When a simulated vehicle or pedestrian interrupts the barrier threshold during downward travel, motion halts in $<80\text{ ms}$ to prevent impact trauma.

---

### Screenshot 14: Gate Operating State — LOCKDOWN (Emergency Protocol)
- **Relative Path:** `docs/report_assets/frontend_screenshots/14_gate_state_lockdown.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 5.8: High-Security Emergency Protocol: gate operating under active `LOCKDOWN` mode (`Lockdown: active`), immobilizing barrier actuation and rejecting all automated biometric/ALPR credentials.*
- **Technical Description:** Depicts the emergency state machine override. Under facility threat conditions, the system overrides normal access logic, logs critical incident telemetry, and locks all entry portals.

---

### Screenshot 15: Diagnostics & Component Health Test Center
- **Relative Path:** `docs/report_assets/frontend_screenshots/15_test_center_diagnostics.png`
- **Resolution:** $1920 \times 1080$ PNG
- **Suggested FYP Report Caption:**  
  > *Figure 6.1: IntelliGate Automated Test Center providing isolated component verification (YuNet face detection, SFace encryption, ALPR registry lookup, CLAHE low-light enhancement, and visit deduplication).*
- **Technical Description:** Illustrates the integrated diagnostic suite. Allows evaluators and engineers to execute end-to-end component health checks in a separate sandbox without corrupting live operational tables.

---

## 4. LaTeX and Word Report Integration Guide

### LaTeX Snippet Example
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{report_assets/frontend_screenshots/02_main_dashboard.png}
    \caption{IntelliGate Central Security Operations Dashboard presenting real-time system health telemetry, biometric/vehicle registry counts, open security alerts, and live event feed.}
    \label{fig:main_dashboard}
\end{figure}
```

### Word Document Best Practices
1. Insert images using **Insert > Pictures > This Device** selecting the high-resolution PNGs directly from `docs/report_assets/frontend_screenshots/`.
2. Set image wrapping to **In Line with Text** with a uniform horizontal width of **6.5 inches** (16.5 cm) to preserve aspect ratio and maintain margin alignment.
3. Apply standard report caption styling directly underneath each figure using the formatted captions provided in Section 3 above.
