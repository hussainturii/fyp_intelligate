"""
IntelliGate — Module 2: Vehicle Recognition & Re-Identification
================================================================
Developer : Hussain Raza (SP23-BAI-057)
Supervisor: Dr. Sameera Batool
Project   : COMSATS University Islamabad FYP 2023-2027

Features:
    FE-1  Detect and recognize vehicle license plates using OCR (EasyOCR)
    FE-2  Perform vehicle re-identification by visual appearance using ViTs
    FE-3  Extract deep-learning feature embeddings for vehicle matching
    FE-4  Track recurring vehicles even when plates are obscured
    FE-5  Maintain vehicle database with access history and patterns
"""

from .vehicle_detector import VehicleDetector, DetectedVehicle
from .plate_detector import PlateDetector, DetectedPlate
from .plate_ocr import PlateOCR, OCRResult
from .vehicle_reid import VehicleReID
from .database.vehicle_db import VehicleDatabase

__all__ = [
    "VehicleDetector", "DetectedVehicle",
    "PlateDetector", "DetectedPlate",
    "PlateOCR", "OCRResult",
    "VehicleReID",
    "VehicleDatabase",
]
__version__ = "1.0.0"
