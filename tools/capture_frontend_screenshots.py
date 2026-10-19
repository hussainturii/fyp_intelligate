"""
Capture professional, high-resolution 1920x1080 screenshots of the running IntelliGate frontend
using Playwright and the native Google Chrome browser.
Captures all functional modules and operating states:
- 01_login.png
- 02_main_dashboard.png
- 03_face_recognition.png
- 04_vehicle_recognition.png
- 05_virtual_gate_controls.png
- 06_camera_replay_monitoring.png
- 07_alerts_security_stranger.png
- 08_queue_scene_monitoring.png
- 09_event_history_audit.png
- 10_reporting_evidence_analytics.png
- 11_gate_state_allow_opening.png
- 12_gate_state_review_required.png
- 13_gate_state_stopped_obstruction.png
- 14_gate_state_lockdown.png
- 15_test_center_diagnostics.png
"""

import json
from pathlib import Path
import time
import urllib.request
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("docs/report_assets/frontend_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://127.0.0.1:8000"


def api_post(endpoint, data=None):
    url = f"{BASE_URL}/api{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"API {endpoint} error: {e}")
        return None


def run_capture():
    print("=== STARTING PROFESSIONAL FRONTEND SCREENSHOT CAPTURE ===")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--disable-gpu", "--no-sandbox"]
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.0
        )
        page = context.new_page()

        # -------------------------------------------------------------
        # 1. Login Page (Unauthenticated)
        # -------------------------------------------------------------
        print("\n[1/15] Capturing 01_login.png...")
        page.goto(f"{BASE_URL}/")
        page.wait_for_selector("text=Sign in to the security dashboard")
        time.sleep(1.0)
        page.screenshot(path=str(OUTPUT_DIR / "01_login.png"))
        print("  -> Saved 01_login.png")

        # -------------------------------------------------------------
        # Authenticate as sec_admin
        # -------------------------------------------------------------
        print("\nAuthenticating session as sec_admin...")
        page.fill("input[name='username']", "sec_admin")
        page.fill("input[name='password']", "admin-password-123")
        page.click("button:has-text('Sign in')")
        page.wait_for_selector("text=Overview")
        time.sleep(1.5)

        # -------------------------------------------------------------
        # 2. Main Dashboard Overview
        # -------------------------------------------------------------
        print("\n[2/15] Capturing 02_main_dashboard.png...")
        page.goto(f"{BASE_URL}/")
        page.wait_for_selector("h1:has-text('Overview')")
        page.wait_for_selector("text=System Health")
        time.sleep(2.0)
        page.screenshot(path=str(OUTPUT_DIR / "02_main_dashboard.png"))
        print("  -> Saved 02_main_dashboard.png")

        # -------------------------------------------------------------
        # 3. Face Enrolment & Database
        # -------------------------------------------------------------
        print("\n[3/15] Capturing 03_face_recognition.png...")
        page.goto(f"{BASE_URL}/faces")
        page.wait_for_selector("h1:has-text('Face enrolment')")
        page.wait_for_selector("text=Dr. Tariq Mahmood")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "03_face_recognition.png"))
        print("  -> Saved 03_face_recognition.png")

        # -------------------------------------------------------------
        # 4. Vehicle Database
        # -------------------------------------------------------------
        print("\n[4/15] Capturing 04_vehicle_recognition.png...")
        page.goto(f"{BASE_URL}/vehicles")
        page.wait_for_selector("h1:has-text('Vehicle Database')")
        page.wait_for_selector("text=LEA1234")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "04_vehicle_recognition.png"))
        print("  -> Saved 04_vehicle_recognition.png")

        # -------------------------------------------------------------
        # 5. Virtual Gate Controls (Standard Closed State)
        # -------------------------------------------------------------
        print("\n[5/15] Capturing 05_virtual_gate_controls.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("h1:has-text('Software operations')")
        time.sleep(1.0)
        # Click RESET to ensure clean CLOSED state
        page.click("button:has-text('RESET')")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "05_virtual_gate_controls.png"))
        print("  -> Saved 05_virtual_gate_controls.png")

        # -------------------------------------------------------------
        # 6. Camera / Replay Monitoring
        # -------------------------------------------------------------
        print("\n[6/15] Capturing 06_camera_replay_monitoring.png...")
        page.goto(f"{BASE_URL}/cameras")
        page.wait_for_selector("h1:has-text('Live camera and video replay')")
        time.sleep(1.0)
        try:
            demo_sel = page.locator("select:has(option[value='face_smoke'])")
            if demo_sel.count() > 0:
                demo_sel.select_option("face_smoke")
                time.sleep(0.5)
                page.click("button:has-text('Replay sample in selected mode')")
                time.sleep(3.0)
        except Exception as e:
            print(f"  Note: Demo replay notice: {e}")
        page.screenshot(path=str(OUTPUT_DIR / "06_camera_replay_monitoring.png"))
        print("  -> Saved 06_camera_replay_monitoring.png")

        # -------------------------------------------------------------
        # 7. Security Alerts Center
        # -------------------------------------------------------------
        print("\n[7/15] Capturing 07_alerts_security_stranger.png...")
        page.goto(f"{BASE_URL}/alerts")
        page.wait_for_selector("h1:has-text('Security Alerts')")
        page.wait_for_selector("text=Safety Interlock")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "07_alerts_security_stranger.png"))
        print("  -> Saved 07_alerts_security_stranger.png")

        # -------------------------------------------------------------
        # 8. Queue & Scene Monitoring (Operations Scrolled)
        # -------------------------------------------------------------
        print("\n[8/15] Capturing 08_queue_scene_monitoring.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Scene rules and queue")
        time.sleep(1.0)
        # Scroll main container down to reveal Scene rules, queue config, and stranger visits
        page.evaluate("document.querySelector('main').scrollTop = 380")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "08_queue_scene_monitoring.png"))
        print("  -> Saved 08_queue_scene_monitoring.png")

        # -------------------------------------------------------------
        # 9. Event Log Audit Trail
        # -------------------------------------------------------------
        print("\n[9/15] Capturing 09_event_history_audit.png...")
        page.goto(f"{BASE_URL}/events")
        page.wait_for_selector("h1:has-text('Event Log')")
        page.wait_for_selector("text=exact_plate")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "09_event_history_audit.png"))
        print("  -> Saved 09_event_history_audit.png")

        # -------------------------------------------------------------
        # 10. Reporting & Evidence Analytics
        # -------------------------------------------------------------
        print("\n[10/15] Capturing 10_reporting_evidence_analytics.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Evidence report")
        time.sleep(1.0)
        # Click Generate last 24 hours report button
        page.click("button:has-text('Generate last 24 hours')")
        time.sleep(1.5)
        # Fill in natural language event query search
        page.fill("input[placeholder*='face today']", "authorized vehicle")
        page.click("button:has-text('Search events')")
        time.sleep(1.5)
        # Scroll main container down to reveal evidence report section and search results
        page.evaluate("document.querySelector('main').scrollTop = 650")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "10_reporting_evidence_analytics.png"))
        print("  -> Saved 10_reporting_evidence_analytics.png")

        # -------------------------------------------------------------
        # 11. Gate State: ALLOW / OPENING / OPEN
        # -------------------------------------------------------------
        print("\n[11/15] Capturing 11_gate_state_allow_opening.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Virtual gate")
        # Click RESET first to ensure clean state
        page.click("button:has-text('RESET')")
        time.sleep(0.5)
        # Click OPEN command
        page.click("button:has-text('OPEN')")
        time.sleep(1.2)
        page.screenshot(path=str(OUTPUT_DIR / "11_gate_state_allow_opening.png"))
        print("  -> Saved 11_gate_state_allow_opening.png")

        # -------------------------------------------------------------
        # 12. Gate State: REVIEW_REQUIRED (Access Review Card)
        # -------------------------------------------------------------
        print("\n[12/15] Capturing 12_gate_state_review_required.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Access review")
        page.wait_for_selector("text=Dr. Sarah Qureshi")
        # Ensure gate is reset to CLOSED waiting for operator review
        page.click("button:has-text('RESET')")
        time.sleep(1.5)
        page.screenshot(path=str(OUTPUT_DIR / "12_gate_state_review_required.png"))
        print("  -> Saved 12_gate_state_review_required.png")

        # -------------------------------------------------------------
        # 13. Gate State: STOPPED (Obstruction Active)
        # -------------------------------------------------------------
        print("\n[13/15] Capturing 13_gate_state_stopped_obstruction.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Virtual gate")
        # Simulate obstruction
        page.click("button:has-text('Simulate obstruction')")
        time.sleep(0.5)
        # Click STOP
        page.click("button:has-text('STOP')")
        time.sleep(1.2)
        page.screenshot(path=str(OUTPUT_DIR / "13_gate_state_stopped_obstruction.png"))
        print("  -> Saved 13_gate_state_stopped_obstruction.png")

        # -------------------------------------------------------------
        # 14. Gate State: LOCKDOWN
        # -------------------------------------------------------------
        print("\n[14/15] Capturing 14_gate_state_lockdown.png...")
        page.goto(f"{BASE_URL}/operations")
        page.wait_for_selector("text=Virtual gate")
        # Click LOCKDOWN
        page.click("button:has-text('LOCKDOWN')")
        time.sleep(1.2)
        page.screenshot(path=str(OUTPUT_DIR / "14_gate_state_lockdown.png"))
        print("  -> Saved 14_gate_state_lockdown.png")
        # Reset back to CLOSED so system is left clean
        page.click("button:has-text('Clear simulated obstruction')")
        time.sleep(0.5)
        page.click("button:has-text('RESET')")
        time.sleep(0.5)

        # -------------------------------------------------------------
        # 15. Test Center (Component Checks & Diagnostics)
        # -------------------------------------------------------------
        print("\n[15/15] Capturing 15_test_center_diagnostics.png...")
        page.goto(f"{BASE_URL}/testing")
        page.wait_for_selector("h1:has-text('Test Center')")
        time.sleep(0.5)
        # Click Run component checks
        page.click("button:has-text('Run component checks')")
        # Wait for checks table to complete
        time.sleep(3.5)
        page.screenshot(path=str(OUTPUT_DIR / "15_test_center_diagnostics.png"))
        print("  -> Saved 15_test_center_diagnostics.png")

        browser.close()

    print("\n[COMPLETED] All 15 professional screenshots captured successfully!")


if __name__ == "__main__":
    run_capture()
