"""
utils.py
Helper utilities: drawing, timestamp, cooldown manager, evidence cleanup, clip buffer.
"""

import os
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple
import threading
import shutil

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def current_timestamp() -> float:
    return time.time()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_bboxes(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d.get("label", str(d.get("cls_id", "")))
        color = (0, 0, 255) if label.lower() in ("cell phone", "mobile phone", "phone") else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label}:{d['confidence']:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


class CooldownManager:
    """
    Manage per-event-type cooldowns to avoid spamming backend.
    Usage:
        cd = CooldownManager(default_cooldown=8.0)
        if cd.can_fire('phone_detected'):
            cd.fired('phone_detected')
    """
    def __init__(self, default_cooldown: float = 8.0):
        self.default_cooldown = float(default_cooldown)
        self.last_fired = {}
        self.lock = threading.Lock()

    def can_fire(self, event_type: str) -> bool:
        with self.lock:
            now = time.time()
            last = self.last_fired.get(event_type, 0.0)
            return (now - last) >= self.default_cooldown

    def fired(self, event_type: str):
        with self.lock:
            self.last_fired[event_type] = time.time()


class EvidenceManager:
    """
    Manages short-lived evidence files in evidence_dir.
    - save images
    - cleanup older than ttl_seconds
    - ensure not to keep logs locally long-term
    """
    def __init__(self, evidence_dir: str = "evidence", ttl_seconds: int = 600):
        self.evidence_dir = evidence_dir
        self.ttl_seconds = int(ttl_seconds)
        ensure_dir(self.evidence_dir)

    def save_jpg(self, img: np.ndarray, prefix: str = "evidence") -> str:
        ts = int(time.time() * 1000)
        fname = f"{prefix}_{ts}.jpg"
        path = os.path.join(self.evidence_dir, fname)
        cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(path)
        return path

    def cleanup(self):
        now = time.time()
        for fn in os.listdir(self.evidence_dir):
            path = os.path.join(self.evidence_dir, fn)
            try:
                m = os.path.getmtime(path)
                if now - m > self.ttl_seconds:
                    os.remove(path)
                    logger.info("Removed old evidence: %s", path)
            except Exception:
                logger.exception("Error cleaning %s", path)


class VideoClipBuffer:
    """
    Maintains a rolling buffer of recent frames for saving a short clip around event.
    - buffer_seconds: how many seconds to buffer
    - fps: expected fps
    """
    def __init__(self, buffer_seconds: float = 5.0, fps: int = 10):
        self.fps = int(fps)
        self.max_frames = max(1, int(buffer_seconds * self.fps))
        self.deque = deque(maxlen=self.max_frames)

    def add(self, frame):
        # frame is BGR numpy
        self.deque.append(frame.copy())

    def save_clip(self, out_path: str, codec: str = "mp4v"):
        if not self.deque:
            return False
        h, w = self.deque[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, float(self.fps), (w, h))
        for f in self.deque:
            writer.write(f)
        writer.release()
        return True
