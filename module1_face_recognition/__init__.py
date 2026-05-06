"""
IntelliGate — Module 1: Face Recognition & Anti-Spoofing
=========================================================
Developer : Hussain Raza (SP23-BAI-057)
Supervisor: Dr. Sameera Batool
Project   : COMSATS University Islamabad FYP 2023-2027

Features:
    FE-1  Detect and recognize authorized faces using Vision Transformers (ViTs)
    FE-2  Perform blink detection for liveness verification
    FE-3  Apply texture-based spoof detection to identify photo/video attacks
    FE-4  Implement depth-based liveness cues for 3D face validation
    FE-5  Detect and prevent mask-based spoofing attempts
"""

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .liveness_detector import LivenessDetector
from .database.face_db import FaceDatabase

__all__ = ["FaceDetector", "FaceRecognizer", "LivenessDetector", "FaceDatabase"]
__version__ = "1.0.0"
