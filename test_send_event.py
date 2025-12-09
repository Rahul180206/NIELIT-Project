"""
test_send_event.py
Simple smoke test to POST a sample JSON event to backend.
Usage:
    python test_send_event.py --backend http://localhost:5000/api/event
"""

import argparse
import json
import time
import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:5000/api/event")
    args = p.parse_args()

    sample = {
        "module": "yolo_object_detection",
        "timestamp": time.time(),
        "session_id": None,
        "camera_id": "cam0",
        "detections": [
            {"label": "cell phone", "cls_id": 67, "confidence": 0.91, "bbox": [100,100,200,220]}
        ],
        "summary": {
            "phone_detected": True,
            "person_count": 1,
            "max_confidence": 0.91
        },
        "evidence_image": None
    }
    r = requests.post(args.backend, json=sample)
    print("Response:", r.status_code, r.text)


if __name__ == "__main__":
    main()
