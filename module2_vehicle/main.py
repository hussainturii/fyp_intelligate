"""
main.py — Module 2: Vehicle Recognition & Re-Identification
============================================================
Main pipeline entry point for Module 2.

Full pipeline per frame:
    Camera → VehicleDetector → PlateDetector → PlateOCR
           → VehicleReID → VehicleDatabase lookup → Gate Decision

Decision logic:
    1. Plate found + authorized plate in DB          → GRANT (method: plate)
    2. Plate found but unknown + Re-ID matches DB    → GRANT (method: reid)
    3. Plate obscured + Re-ID matches DB             → GRANT (method: reid)
    4. Neither plate nor Re-ID matches               → DENY, log unknown vehicle

Run modes:
    python -m module2_vehicle.main --mode webcam
    python -m module2_vehicle.main --mode register --plate "ABC-1234" --owner "Hussain"
    python -m module2_vehicle.main --mode test --image path/to/image.jpg

Author: Hussain Raza (SP23-BAI-057)
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

from .database.vehicle_db import VehicleDatabase
from .plate_detector import PlateDetector
from .plate_ocr import PlateOCR
from .vehicle_detector import VehicleDetector
from .vehicle_reid import VehicleReID


# ─────────────────────────────────────────────────────────────────────────────
# Gate Decision Constants
# ─────────────────────────────────────────────────────────────────────────────

class GateDecision:
    GRANT   = "GRANT_ACCESS"
    DENY    = "DENY_ACCESS"
    WAITING = "WAITING"
    UNKNOWN = "UNKNOWN_VEHICLE"


# ─────────────────────────────────────────────────────────────────────────────
# Main Module 2 Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IntelliGateModule2:
    """
    Complete Module 2 runtime pipeline.

    Ties together:
        VehicleDetector → PlateDetector → PlateOCR
        → VehicleReID → VehicleDatabase → GateDecision
    """

    def __init__(self, config_path: str = "config/config.yaml", on_event_callback=None) -> None:
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        hw  = self.cfg["hardware"]

        # Resolve device
        requested = hw["device"].lower()
        self.device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and self.device == "cpu":
            logger.warning("[Module2] CUDA not available — falling back to CPU.")

        logger.info("[Module2] Initializing Vehicle Recognition pipeline...")

        self.vehicle_det = VehicleDetector(
            model_name="yolov8n.pt",
            confidence_threshold=0.40,
            device=self.device,
            use_tracker=True,           # persistent IDs across frames
        )

        self.plate_det = PlateDetector(
            confidence_threshold=0.35,
            device=self.device,
        )

        self.plate_ocr = PlateOCR(
            languages=["en"],
            gpu=(self.device == "cuda"),
            n_reads=3,
            min_confidence=0.55,
        )

        self.reid = VehicleReID(
            similarity_threshold=0.82,
            device=self.device,
        )

        self.db = VehicleDatabase(
            db_path="module2_vehicle/database/vehicles.h5",
            unknown_db_path="module2_vehicle/database/unknown_vehicles.h5",
            embedding_dim=768,
            reid_threshold=0.82,
            fuzzy_threshold=0.80,
        )

        self._frame_count = 0
        self._stats = {"total": 0, "granted": 0, "denied": 0, "reid_saves": 0}
        self.on_event_callback = on_event_callback
        self._last_logged_time = {}
        self.cooldown_seconds = 10.0
        
        # Vehicle-level deduplication: embedding hash → last_seen_time
        # Prevents same physical vehicle from being logged multiple times within cooldown
        self._vehicle_embedding_cache = {}
        self._vehicle_cache_cooldown = 10.0

        logger.success("[Module2] Pipeline ready.")

    def _should_log(self, event_key: str) -> bool:
        now = time.time()
        if now - self._last_logged_time.get(event_key, 0.0) >= self.cooldown_seconds:
            self._last_logged_time[event_key] = now
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Frame Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one BGR frame through the full vehicle pipeline.

        Returns:
            dict with:
                decision         (str)
                plate_text       (str)
                owner_name       (str)
                confidence       (float)
                method           (str) 'exact_plate'|'fuzzy_plate'|'reid'
                vehicles_found   (int)
                alerts           (list)
                annotated_frame  (ndarray)
        """
        self._frame_count += 1
        result = {
            "decision":        GateDecision.WAITING,
            "plate_text":      "",
            "owner_name":      "",
            "vehicle_info":    {},
            "confidence":      0.0,
            "method":          "",
            "vehicles_found":  0,
            "alerts":          [],
            "annotated_frame": frame_bgr.copy(),
        }

        # ── Step 1: Detect vehicles ──────────────────────────────────────────
        vehicles = self.vehicle_det.detect(frame_bgr)
        result["vehicles_found"] = len(vehicles)

        if not vehicles:
            return result

        self._stats["total"] += 1
        vehicle = self.vehicle_det.get_primary_vehicle(vehicles)

        # ── Step 2: Detect license plate in vehicle crop ──────────────────────
        ox, oy = vehicle.bbox[0], vehicle.bbox[1]
        plates = self.plate_det.detect(vehicle.crop, offset=(ox, oy))

        plate_text = ""
        plate_conf = 0.0

        if plates:
            best_plate = plates[0]
            # ── Step 3: OCR ──────────────────────────────────────────────────
            ocr_result = self.plate_ocr.read(best_plate.crop)
            if ocr_result.confidence >= 0.55:
                plate_text = ocr_result.clean_text
                plate_conf = ocr_result.confidence
                result["plate_text"] = plate_text
                logger.debug(
                    f"[Module2] OCR: '{plate_text}' "
                    f"(conf={plate_conf:.2f}, valid={ocr_result.is_valid})"
                )

        # ── Step 4: Re-ID embedding ───────────────────────────────────────────
        vehicle_pil = Image.fromarray(
            cv2.cvtColor(vehicle.crop, cv2.COLOR_BGR2RGB)
        )
        embedding = self.reid.extract_embedding(vehicle_pil)

        # ── Step 5: Database lookup (plate → fuzzy plate → Re-ID) ─────────────
        db_result = self.db.combined_lookup(
            plate=plate_text if plate_text else None,
            embedding=embedding,
        )

        # ── Step 6: Gate decision ─────────────────────────────────────────────
        if db_result:
            # Authorized vehicle found
            self._stats["granted"] += 1
            vid = db_result["vehicle_id"]
            self.db.log_access(vid)

            result["decision"]    = GateDecision.GRANT
            result["owner_name"]  = db_result["owner_name"]
            result["vehicle_info"] = db_result
            result["confidence"]  = db_result.get("similarity", 1.0)
            result["method"]      = db_result.get("method", "unknown")

            if self.on_event_callback and self._should_log(db_result.get("owner_name", "UNKNOWN")):
                self.on_event_callback({
                    "module": "vehicle",
                    "event_type": "access_granted",
                    "identity": db_result.get("owner_name", "UNKNOWN"),
                    "confidence": float(result["confidence"]),
                    "method": result["method"]
                })

            logger.success(
                f"[Module2] GRANT: {db_result['plate']} "
                f"({db_result['owner_name']}) via {result['method']}"
            )
        else:
            # Unknown vehicle — apply embedding-level deduplication
            self._stats["denied"] += 1
            result["decision"] = GateDecision.DENY

            # Check if this exact embedding was seen recently (same physical vehicle)
            embedding_hash = hash(embedding.tobytes())
            now = time.time()
            last_seen = self._vehicle_embedding_cache.get(embedding_hash, 0.0)
            
            # Only log if enough time has passed since last identical embedding
            should_log_unknown = (now - last_seen) >= self._vehicle_cache_cooldown
            
            if should_log_unknown:
                self._vehicle_embedding_cache[embedding_hash] = now
                unknown_info = self.db.log_unknown(
                    embedding=embedding, plate=plate_text
                )
            else:
                # Skip logging, reuse last info for this embedding
                unknown_info = {"should_alert": False, "unknown_id": "", "visit_count": 0}

            if self.on_event_callback and should_log_unknown and self._should_log(plate_text if plate_text else "UNKNOWN_VEHICLE"):
                self.on_event_callback({
                    "module": "vehicle",
                    "event_type": "stranger_alert",
                    "identity": plate_text if plate_text else "UNKNOWN_VEHICLE",
                    "confidence": float(plate_conf),
                    "method": "YOLO/OCR" if plate_text else "Re-ID"
                })

            if unknown_info["should_alert"]:
                result["alerts"].append({
                    "type":        "repeat_unknown_vehicle",
                    "unknown_id":  unknown_info["unknown_id"],
                    "visit_count": unknown_info["visit_count"],
                    "plate":       plate_text or "N/A",
                    "message": (
                        f"Repeat unknown vehicle! "
                        f"Visit #{unknown_info['visit_count']} "
                        f"(plate: {plate_text or 'unreadable'})"
                    ),
                })
                logger.warning(
                    f"[Module2] REPEAT UNKNOWN: "
                    f"{unknown_info['visit_count']}× "
                    f"plate='{plate_text or 'N/A'}'"
                )
            else:
                logger.info(
                    f"[Module2] DENY: Unknown vehicle "
                    f"(plate='{plate_text or 'N/A'}')"
                )

        # ── Step 7: Annotate frame ────────────────────────────────────────────
        annotated = self.vehicle_det.draw_detections(frame_bgr, vehicles)
        if plates:
            annotated = self.plate_det.draw_plates(annotated, plates)
        annotated = self._draw_decision_banner(annotated, result)
        result["annotated_frame"] = annotated

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_vehicle(
        self,
        plate:      str,
        owner_name: str,
        owner_id:   str = "",
        make:       str = "",
        model:      str = "",
        color:      str = "",
        image_path: Optional[str] = None,
        capture_reid: bool = True,
    ) -> str:
        """
        Register an authorized vehicle.

        Optionally captures a Re-ID embedding from a photo or webcam
        so the vehicle can be identified even if the plate is obscured.

        Args:
            plate       : License plate string e.g. "ABC-1234"
            owner_name  : Owner's name
            owner_id    : Face DB user_id to link (optional)
            make/model/color : Vehicle metadata
            image_path  : Path to vehicle photo for Re-ID, or None for webcam
            capture_reid: Whether to capture Re-ID embedding

        Returns:
            vehicle_id
        """
        plate = plate.upper().strip()
        embedding = None

        if capture_reid:
            if image_path:
                frame = cv2.imread(image_path)
                if frame is not None:
                    vehicles = self.vehicle_det.detect(frame)
                    if vehicles:
                        v_pil = Image.fromarray(
                            cv2.cvtColor(vehicles[0].crop, cv2.COLOR_BGR2RGB)
                        )
                        embedding = self.reid.extract_embedding(v_pil)
                        logger.info("[Module2] Re-ID embedding captured from image.")
                    else:
                        logger.warning("[Module2] No vehicle detected in image. Registering plate-only.")
                else:
                    logger.warning(f"[Module2] Cannot read image: {image_path}")
            else:
                # Webcam capture
                logger.info("[Module2] Point camera at the vehicle. Press SPACE to capture, Q to skip.")
                cap = cv2.VideoCapture(self.cfg["hardware"]["camera_index"])
                embeddings = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    vehicles = self.vehicle_det.detect(frame)
                    vis = self.vehicle_det.draw_detections(frame, vehicles)
                    cv2.putText(vis, "SPACE=capture  Q=skip", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Register Vehicle", vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' ') and vehicles:
                        v = vehicles[0]
                        v_pil = Image.fromarray(cv2.cvtColor(v.crop, cv2.COLOR_BGR2RGB))
                        embeddings.append(self.reid.extract_embedding(v_pil))
                        logger.info(f"[Module2] Captured embedding {len(embeddings)}/5")
                        if len(embeddings) >= 5:
                            break
                    elif key == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

                if embeddings:
                    avg = np.mean(embeddings, axis=0)
                    embedding = avg / np.linalg.norm(avg)
                    logger.info("[Module2] Re-ID embedding averaged from 5 captures.")

        vid = self.db.register(
            plate=plate,
            owner_name=owner_name,
            owner_id=owner_id,
            make=make, model=model, color=color,
            embedding=embedding,
        )
        logger.success(f"[Module2] Registered {plate} ({owner_name}) → {vid}")
        return vid

    # ─────────────────────────────────────────────────────────────────────────
    # Webcam Demo
    # ─────────────────────────────────────────────────────────────────────────

    def run_webcam(self) -> None:
        """Run the full Module 2 pipeline on webcam."""
        hw  = self.cfg["hardware"]
        cap = cv2.VideoCapture(hw["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  hw["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hw["camera_height"])
        cap.set(cv2.CAP_PROP_FPS,          hw["fps"])

        logger.info("[Module2] Webcam running. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0     = time.time()
            result = self.process_frame(frame)
            ms     = (time.time() - t0) * 1000

            vis = result["annotated_frame"]
            cv2.putText(vis, f"Latency: {ms:.0f}ms | DB: {len(self.db._vehicles)} vehicles",
                        (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("IntelliGate — Module 2: Vehicle Recognition & Re-ID", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._print_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_decision_banner(
        self, frame: np.ndarray, result: dict
    ) -> np.ndarray:
        decision = result["decision"]
        colors = {
            GateDecision.GRANT:   (0, 255, 0),
            GateDecision.DENY:    (0, 0, 255),
            GateDecision.UNKNOWN: (0, 165, 255),
            GateDecision.WAITING: (255, 255, 0),
        }
        color = colors.get(decision, (255, 255, 255))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)

        label = decision
        if result["plate_text"]:
            label += f"  |  {result['plate_text']}"
        if result["owner_name"]:
            label += f"  |  {result['owner_name']}"
        if result["method"]:
            label += f"  [{result['method']}]"

        cv2.putText(frame, label, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        return frame

    def _print_stats(self) -> None:
        logger.info(
            f"[Module2] Stats — Total: {self._stats['total']} | "
            f"Granted: {self._stats['granted']} | "
            f"Denied: {self._stats['denied']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntelliGate Module 2 — Vehicle Recognition & Re-ID"
    )
    parser.add_argument("--mode",   choices=["webcam", "register", "test"], default="webcam")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--plate",  default="",        help="Plate for register mode")
    parser.add_argument("--owner",  default="Unknown", help="Owner name for register mode")
    parser.add_argument("--make",   default="",        help="Vehicle make")
    parser.add_argument("--model",  default="",        help="Vehicle model")
    parser.add_argument("--color",  default="",        help="Vehicle color")
    parser.add_argument("--image",  default=None,      help="Image path for test/register")
    args = parser.parse_args()

    pipeline = IntelliGateModule2(config_path=args.config)

    if args.mode == "webcam":
        pipeline.run_webcam()

    elif args.mode == "register":
        if not args.plate:
            print("Provide --plate for register mode. Example: --plate ABC-1234")
            return
        vid = pipeline.register_vehicle(
            plate=args.plate,
            owner_name=args.owner,
            make=args.make,
            model=args.model,
            color=args.color,
            image_path=args.image,
        )
        print(f"Registered! Vehicle ID: {vid}")

    elif args.mode == "test":
        if not args.image:
            print("Provide --image for test mode.")
            return
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read: {args.image}")
            return
        result = pipeline.process_frame(frame)
        print(f"Decision   : {result['decision']}")
        print(f"Plate      : {result['plate_text']}")
        print(f"Owner      : {result['owner_name']}")
        print(f"Method     : {result['method']}")
        print(f"Confidence : {result['confidence']:.3f}")
        cv2.imshow("Module 2 Result", result["annotated_frame"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
