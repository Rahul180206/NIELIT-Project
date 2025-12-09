"""
detector.py

YOLODetector - lightweight wrapper around ultralytics YOLO model.

Features:
- Auto device selection ('auto' -> CUDA if available else CPU)
- Confidence thresholding
- Restrict to a subset of classes (by name)
- Annotated frame output (optional)
- Helper: is_suspicious(detections) - checks if any detection is in suspicious_objects

Usage:
    from detector import YOLODetector
    det = YOLODetector("yolov8n.pt", device="auto", conf_threshold=0.5,
                       classes=["person","cell phone","book","laptop"])
    dets, annotated = det.detect(frame)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLODetector: wraps ultralytics YOLO model.

    Args:
        model_path: path to .pt weights (default: yolov8n.pt)
        device: 'auto', 'cpu' or 'cuda'
        conf_threshold: confidence cutoff for detections (0-1)
        classes: list of class names to keep (None = all)
        imgsz: inference image size for model (default 640)
    """

    # default suspicious objects (expanded)
    DEFAULT_SUSPICIOUS = {
        "cell phone",
        "book",
        "laptop",
        "keyboard",
        "mouse",
        "remote",
        "backpack",
        "handbag",
        "tv",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.5,
        classes: Optional[List[str]] = None,
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)

        # device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Initializing YOLODetector: device=%s model=%s", self.device, self.model_path)

        # load model
        self.model = YOLO(self.model_path)

        # try to move to cuda if requested
        try:
            if self.device == "cuda":
                self.model.to("cuda")
        except Exception as e:
            logger.warning("Could not move model to CUDA: %s", e)
            self.device = "cpu"

        # build names mapping
        self.names = self.model.names if hasattr(self.model, "names") else {}
        if isinstance(self.names, list):
            self.names = {i: n for i, n in enumerate(self.names)}

        # set class filter (by numeric id)
        self.class_mask = None
        if classes:
            inv = {v: k for k, v in self.names.items()}
            mask = set()
            for cname in classes:
                if cname in inv:
                    mask.add(inv[cname])
                else:
                    # warn but continue - user might add custom classes later
                    logger.warning("Requested class '%s' not in model names", cname)
            if mask:
                self.class_mask = mask

        # suspicious objects set (string labels). Can be overridden externally.
        self.suspicious_objects = set(self.DEFAULT_SUSPICIOUS)

    def detect(self, frame: np.ndarray, annotate: bool = True) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Run detection on a BGR frame and return:
          - detections: list of dicts {label, cls_id, confidence, bbox=[x1,y1,x2,y2]}
          - annotated_frame (BGR numpy) if annotate=True else None

        Note: uses ultralytics YOLO API.
        """
        # run model inference
        results = self.model(frame, imgsz=self.imgsz, verbose=False)[0]

        detections: List[Dict] = []
        for box in results.boxes:
            # robust extraction: torch tensor or numpy
            try:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cid = int(box.cls[0].cpu().numpy())
            except Exception:
                # fallback for older ultralytics versions
                xyxy = box.xyxy[0].numpy()
                conf = float(box.conf[0].numpy())
                cid = int(box.cls[0].numpy())

            if conf < self.conf_threshold:
                continue
            if self.class_mask is not None and cid not in self.class_mask:
                continue

            x1, y1, x2, y2 = map(int, map(float, xyxy))
            label = self.names.get(cid, str(cid))
            detections.append(
                {
                    "label": label,
                    "cls_id": int(cid),
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        annotated = None
        if annotate:
            annotated = self._annotate_frame(frame, detections)

        return detections, annotated

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of frame and return it."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            lbl = d["label"]
            conf = d["confidence"]
            # red for suspicious, green otherwise
            if lbl.lower() in self.suspicious_objects:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{lbl} {conf:.2f}"
            cv2.putText(out, text, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

    def is_suspicious(self, detections: List[Dict]) -> bool:
        """Return True if any detection label is in the suspicious objects set."""
        for d in detections:
            if d["label"].lower() in self.suspicious_objects:
                return True
        return False
