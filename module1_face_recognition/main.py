"""
main.py — Module 1: Face Recognition & Anti-Spoofing
======================================================
Main entry point for Module 1 pipeline.

Full pipeline per frame:
    Camera → FaceDetector → LivenessDetector → FaceRecognizer → Decision

Gate Decision Logic (from SRS Business Rules):
    BR-01: Access ONLY if liveness_score >= 0.90
    BR-02: Spoofing attempts logged with snapshot
    BR-07: Stranger alert if same unknown face 3+ times in 24h
    BR-10: Admin pre-registers all authorized faces

Run modes:
    python main.py --mode webcam       # Live webcam demo
    python main.py --mode register     # Register a new face (admin)
    python main.py --mode test         # Test on a static image

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

from .database.face_db import FaceDatabase
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .liveness_detector import LivenessDetector


# ─────────────────────────────────────────────────────────────────────────────
# Gate Decision
# ─────────────────────────────────────────────────────────────────────────────

class GateDecision:
    GRANT  = "GRANT_ACCESS"
    DENY   = "DENY_ACCESS"
    SPOOF  = "SPOOF_DETECTED"
    MASK   = "MASK_DETECTED"
    WAITING = "WAITING"


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IntelliGateModule1:
    """
    Complete Module 1 runtime pipeline.

    Ties together:
        FaceDetector → LivenessDetector → FaceRecognizer → GateDecision
    """

    def __init__(self, config_path: str = "config/config.yaml", on_event_callback=None) -> None:
        # Load config
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        fc = self.cfg["face_recognition"]
        as_cfg = self.cfg["anti_spoofing"]
        hw = self.cfg["hardware"]

        requested_device = hw["device"].lower()
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and self.device == "cpu":
            logger.warning(
                "[Module1] CUDA requested in config but torch.cuda.is_available() is False. Falling back to CPU."
            )

        logger.info("[Module1] Initializing IntelliGate Face Recognition pipeline...")

        # Initialize components
        self.detector = FaceDetector(
            device=self.device,
            min_face_size=fc["detection"]["min_face_size"],
            thresholds=tuple(fc["detection"]["thresholds"]),
            margin=fc["detection"]["margin"],
            image_size=fc["recognition"]["image_size"],
        )

        self.recognizer = FaceRecognizer(
            model_name=fc["recognition"]["model"],
            similarity_threshold=fc["recognition"]["similarity_threshold"],
            device=self.device,
        )

        self.liveness = LivenessDetector(
            liveness_threshold=as_cfg["liveness_threshold"],
            weights=as_cfg["weights"],
            device=self.device,
            blink_required=as_cfg["blink"]["required_blinks"],
            blink_window_frames=as_cfg["blink"]["window_frames"],
        )

        self.db = FaceDatabase(
            authorized_db_path=fc["database"]["path"],
            stranger_db_path=fc["database"]["stranger_db_path"],
            embedding_dim=fc["recognition"]["embedding_dim"],
            stranger_retention_days=fc["database"]["stranger_retention_days"],
            repeat_visit_threshold=fc["database"]["repeat_visit_threshold"],
            repeat_visit_window_hours=fc["database"]["repeat_visit_window_hours"],
        )

        self.liveness_sessions: dict = {}   # track_id → LivenessSession
        self._frame_count = 0
        self._stats = {
            "total_faces": 0, "authorized": 0,
            "unknown": 0, "spoofs": 0,
        }
        self.on_event_callback = on_event_callback
        self._last_logged_time = {}
        self.cooldown_seconds = 10.0

        logger.success("[Module1] Pipeline ready.")

    def _should_log(self, event_key: str) -> bool:
        now = time.time()
        if now - self._last_logged_time.get(event_key, 0.0) >= self.cooldown_seconds:
            self._last_logged_time[event_key] = now
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main Processing Loop
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one BGR video frame through the full pipeline.

        Returns:
            dict with decision, identity, confidence, annotations (frame copy)
        """
        self._frame_count += 1
        result = {
            "decision": GateDecision.WAITING,
            "identity": None,
            "confidence": 0.0,
            "liveness_score": 0.0,
            "faces_detected": 0,
            "annotated_frame": frame_bgr.copy(),
            "alerts": [],
        }

        # Step 1: Detect faces
        faces = self.detector.detect(frame_bgr)
        result["faces_detected"] = len(faces)

        if not faces:
            return result

        # Process the highest-confidence face (primary subject at gate)
        face = self.detector.get_largest_face(faces)
        self._stats["total_faces"] += 1

        # Step 2: Get or create liveness session for this face
        # Simple tracking: use a fixed session key for the primary face
        # (In production, use a proper face tracker like SORT/ByteTrack)
        session_key = "primary"
        if session_key not in self.liveness_sessions:
            self.liveness_sessions[session_key] = self.liveness.new_session()

        session = self.liveness_sessions[session_key]

        # Step 3: Run liveness check
        live_result = self.liveness.process_frame(
            frame_bgr, face.aligned_pil, session
        )
        result["liveness_score"] = live_result["fused_score"]

        if not live_result["decision_made"]:
            # Still collecting evidence
            annotated = self.detector.draw_detections(frame_bgr, [face])
            annotated = self.liveness.draw_liveness_overlay(annotated, session)
            result["annotated_frame"] = annotated
            return result

        # Step 4: Liveness decision made
        if not live_result["is_live"]:
            # Spoof detected
            self._stats["spoofs"] += 1
            result["decision"] = GateDecision.SPOOF
            result["alerts"].append({
                "type": "spoof",
                "reason": live_result["deny_reason"],
                "snapshot": face.aligned_face,
            })
            logger.warning(f"[Module1] SPOOF DENIED: {live_result['deny_reason']}")

            # Log spoof to DB (BR-02)
            self._log_spoof_event(face, live_result)

            if self.on_event_callback and self._should_log("SPOOF"):
                self.on_event_callback({
                    "module": "face",
                    "event_type": "spoof_detected",
                    "identity": "UNKNOWN",
                    "confidence": float(result["liveness_score"]),
                    "method": "liveness"
                })

            # Reset session for next person
            del self.liveness_sessions[session_key]

        else:
            # Step 5: Liveness PASSED → Recognize
            rec_result = self.recognizer.recognize(face.aligned_pil, self.db)
            result["confidence"] = rec_result.confidence

            if rec_result.is_authorized:
                # Access Granted
                self._stats["authorized"] += 1
                result["decision"]  = GateDecision.GRANT
                result["identity"]  = rec_result.identity
                self.db.log_access(rec_result.user_id)

                if self.on_event_callback and self._should_log(rec_result.identity):
                    self.on_event_callback({
                        "module": "face",
                        "event_type": "access_granted",
                        "identity": rec_result.identity,
                        "confidence": float(rec_result.confidence),
                        "method": "FaceID"
                    })

                logger.success(
                    f"[Module1] ACCESS GRANTED: {rec_result.identity} "
                    f"(sim={rec_result.confidence:.3f})"
                )
            else:
                # Unknown person
                self._stats["unknown"] += 1
                result["decision"] = GateDecision.DENY
                result["identity"] = "UNKNOWN"

                # Log to stranger database (FR-07)
                stranger_info = self.db.add_stranger(rec_result.embedding)

                if self.on_event_callback and self._should_log("UNKNOWN_FACE"):
                    self.on_event_callback({
                        "module": "face",
                        "event_type": "stranger_alert",
                        "identity": "UNKNOWN",
                        "confidence": float(rec_result.confidence),
                        "method": "FaceID"
                    })

                if stranger_info["should_alert"]:
                    result["alerts"].append({
                        "type": "repeat_stranger",
                        "stranger_id": stranger_info["stranger_id"],
                        "visit_count": stranger_info["visit_count"],
                        "message": f"Repeat unknown visitor! Visit #{stranger_info['visit_count']}",
                    })

                logger.info(
                    f"[Module1] ACCESS DENIED: Unknown face "
                    f"(sim={rec_result.confidence:.3f})"
                )

            # Reset session for next person
            del self.liveness_sessions[session_key]

        # Step 6: Annotate frame
        annotated = self.detector.draw_detections(frame_bgr, [face])
        annotated = self._draw_gate_decision(annotated, result)
        result["annotated_frame"] = annotated

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Admin: Face Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_face(
        self,
        name: str,
        role: str,
        image_path: str | None = None,
        n_frames: int = 5,
    ) -> str:
        """
        Register a new authorized face. (Admin only — BR-10)

        Captures N frames and averages embeddings for robustness.

        Args:
            name       : Person's name
            role       : 'resident' | 'employee' | 'admin'
            image_path : Path to image file, or None to use webcam
            n_frames   : Number of frames to average (webcam mode)

        Returns:
            user_id of registered face
        """
        logger.info(f"[Module1] Registering: {name} ({role})")
        embeddings = []

        if image_path:
            # Register from file
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            faces = self.detector.detect(img)
            if not faces:
                raise ValueError("No face detected in registration image.")
            emb = self.recognizer.extract_embedding(faces[0].aligned_pil)
            embeddings.append(emb)
        else:
            # Register from webcam
            cap = cv2.VideoCapture(self.cfg["hardware"]["camera_index"])
            collected = 0
            logger.info(f"[Module1] Look at camera. Collecting {n_frames} frames...")

            while collected < n_frames:
                ret, frame = cap.read()
                if not ret:
                    continue
                faces = self.detector.detect(frame)
                if faces:
                    emb = self.recognizer.extract_embedding(faces[0].aligned_pil)
                    embeddings.append(emb)
                    collected += 1
                    logger.info(f"  Frame {collected}/{n_frames} captured.")
                cv2.imshow("Registration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        if not embeddings:
            raise ValueError("No valid face embeddings collected.")

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)   # Re-normalize

        user_id = self.db.register(name, role, avg_embedding)
        logger.success(f"[Module1] Registered {name} → {user_id}")
        return user_id

    # ─────────────────────────────────────────────────────────────────────────
    # Webcam Demo
    # ─────────────────────────────────────────────────────────────────────────

    def run_webcam(self) -> None:
        """Run the full Module 1 pipeline on the webcam."""
        hw = self.cfg["hardware"]
        cap = cv2.VideoCapture(hw["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, hw["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hw["camera_height"])
        cap.set(cv2.CAP_PROP_FPS, hw["fps"])

        logger.info("[Module1] Starting webcam. Press 'q' to quit, 'r' to reset session.")

        fps_timer = time.time()
        fps_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed.")
                break

            # Process frame
            t0 = time.time()
            result = self.process_frame(frame)
            latency_ms = (time.time() - t0) * 1000

            # FPS counter
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_timer = time.time()
            else:
                fps = fps_count

            # Display
            vis = result["annotated_frame"]
            cv2.putText(vis, f"FPS: {fps} | Latency: {latency_ms:.0f}ms",
                        (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("IntelliGate — Module 1: Face Recognition & Anti-Spoofing", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.liveness_sessions.clear()
                logger.info("[Module1] Session reset.")

        cap.release()
        cv2.destroyAllWindows()
        self._print_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_gate_decision(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw the gate decision on the frame."""
        decision = result["decision"]
        colors = {
            GateDecision.GRANT:   (0, 255, 0),
            GateDecision.DENY:    (0, 165, 255),
            GateDecision.SPOOF:   (0, 0, 255),
            GateDecision.WAITING: (255, 255, 0),
        }
        color = colors.get(decision, (255, 255, 255))

        # Big status banner at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        label = decision
        if result["identity"]:
            label += f" | {result['identity']}"
        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Liveness score bar
        score = result["liveness_score"]
        bar_w = int(frame.shape[1] * score)
        bar_color = (0, 255, 0) if score >= 0.9 else (0, 165, 255)
        cv2.rectangle(frame, (0, 50), (bar_w, 58), bar_color, -1)

        return frame

    def _log_spoof_event(self, face, live_result: dict) -> None:
        """Save spoof snapshot for security audit (BR-02)."""
        try:
            snap_dir = Path("logs/spoof_snapshots")
            snap_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            snap_path = snap_dir / f"spoof_{timestamp}.jpg"
            import cv2
            bgr_face = cv2.cvtColor(face.aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(snap_path), bgr_face)
            logger.info(f"[Module1] Spoof snapshot saved: {snap_path}")
        except Exception as e:
            logger.error(f"[Module1] Failed to save spoof snapshot: {e}")

    def _print_stats(self) -> None:
        logger.info(
            f"[Module1] Session stats — "
            f"Total: {self._stats['total_faces']} | "
            f"Authorized: {self._stats['authorized']} | "
            f"Unknown: {self._stats['unknown']} | "
            f"Spoofs: {self._stats['spoofs']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IntelliGate Module 1 — Face Recognition & Anti-Spoofing")
    parser.add_argument("--mode", choices=["webcam", "register", "test"], default="webcam",
                        help="Run mode")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--name",  default="New User",  help="Name for registration mode")
    parser.add_argument("--role",  default="resident",  help="Role for registration mode")
    parser.add_argument("--image", default=None,        help="Image path for test/register mode")
    args = parser.parse_args()

    pipeline = IntelliGateModule1(config_path=args.config)

    if args.mode == "webcam":
        pipeline.run_webcam()

    elif args.mode == "register":
        user_id = pipeline.register_face(
            name=args.name,
            role=args.role,
            image_path=args.image,
        )
        print(f"Registered! User ID: {user_id}")

    elif args.mode == "test":
        if not args.image:
            print("Provide --image path for test mode.")
            return
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read: {args.image}")
            return
        result = pipeline.process_frame(frame)
        print(f"Decision  : {result['decision']}")
        print(f"Identity  : {result['identity']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Liveness  : {result['liveness_score']:.3f}")
        cv2.imshow("Result", result["annotated_frame"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
