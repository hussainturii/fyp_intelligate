from pathlib import Path
from module8_dashboard.backend.model_integrity import verify_model_file

models = [
    "models/face_detection_yunet_2023mar.onnx",
    "models/face_recognition_sface_2021dec.onnx",
    "models/yolov8n.pt",
    "models/plate_detector_yolov8n.pt",
    "models/2.7_80x80_MiniFASNetV2.pth",
    "models/osnet_ain_x1_0_vehicle_reid.onnx",
]

for m in models:
    res = verify_model_file(m, strict=True)
    print(f"{Path(m).name}: verified={res.get('verified')} ({res.get('status')})")
