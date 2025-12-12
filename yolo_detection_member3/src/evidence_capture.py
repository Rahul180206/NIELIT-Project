# src/evidence_capture.py
import os
import cv2
from datetime import datetime
from src.utils import make_output, save_json
import uuid

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)

def save_evidence_image(frame, prefix='evidence'):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(EVIDENCE_DIR, fname)
    cv2.imwrite(path, frame)
    return path

def handle_event(frame, detections, event_type, frame_idx=None, extra=None):
    img_path = save_evidence_image(frame, prefix=event_type)
    output = make_output(event_type=event_type, boxes=detections, image_path=img_path, frame_idx=frame_idx, extra=extra)
    json_path = save_json(output)
    return img_path, json_path
