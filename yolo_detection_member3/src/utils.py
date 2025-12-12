# src/utils.py
import time
import json
import os
from datetime import datetime
import logging

LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'logs', 'detections.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def make_output(event_type: str, boxes, image_path: str=None, frame_idx:int=None, extra=None):
    """Structured JSON for backend."""
    obj_list = []
    for b in boxes:
        obj_list.append({
            'class': b['class_name'],
            'confidence': float(b['confidence']),
            'box': [float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])]
        })
    out = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'event_type': event_type,
        'frame_idx': frame_idx,
        'image_path': image_path,
        'detections': obj_list,
        'extra': extra or {}
    }
    return out

def save_json(output: dict, outdir='backend'):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"detection_{int(time.time())}.json")
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2)
    logging.info("Saved JSON: %s", fname)
    return fname
