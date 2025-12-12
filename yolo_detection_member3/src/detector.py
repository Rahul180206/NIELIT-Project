import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="models/yolov8n.pt", device="cpu",
                 conf=0.20, iou=0.30, classes=None):

        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes  # ['person', 'cell phone', 'laptop', 'tv', 'book']

        # Load both .pt and .onnx cleanly
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            raise

        # Apply sensitivity settings
        self.model.conf = conf
        self.model.iou = iou

        print(f"[INFO] YOLO model loaded: {model_path}")
        print(f"[INFO] CONF={conf}, IOU={iou}, DEVICE={device}")

    def predict_frame(self, frame):
        """Runs YOLO detection on a single frame."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=rgb,
            imgsz=640,              # optimized for performance
            device=self.device,
            stream=False,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        r = results[0]

        # Extract YOLO detections
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = r.names[cls_id]
            conf = float(box.conf)

            # Apply class filter (if provided)
            if self.classes and cls_name not in self.classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

        return detections
