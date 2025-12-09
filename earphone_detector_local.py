"""
Local earphone detector using trained YOLO model
"""
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)

class LocalEarphoneDetector:
    def __init__(self, model_path: str = r"C:\Users\rahul\runs\detect\earphone_worn_detector2\weights\best.pt", conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logger.info(f"Local earphone detector loaded from {model_path}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect earphones in frame and return list of detections"""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            detections.append({
                "label": "earphone",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        
        return detections
