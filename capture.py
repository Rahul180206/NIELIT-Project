"""
capture.py

Main capture loop for YOLO Proctor (Member 3).

Responsibilities:
- Open webcam / video source
- Run YOLODetector.detect(frame)
- Apply simple suspicious rules (phone detection, multi-person, other suspicious classes)
- Earphone detection via Roboflow Workflow (cloud inference)
- Use per-event cooldowns to avoid spamming
- Save ephemeral evidence images and upload via sender (in background)
- Optionally save a short video clip around the event
- Non-blocking backend calls so detection loop remains responsive

Usage examples:
    # dry run (don't actually POST)
    python capture.py --model yolov8n.pt --backend http://localhost:5000/api/event --dry-run

    # run and send events to local backend
    python capture.py --model yolov8n.pt --backend http://localhost:5000/api/event

    # limit FPS + enable clip saving
    python capture.py --model yolov8n.pt --fps 10 --save-video --clip-seconds 6
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import threading
from typing import List, Optional

import cv2
import numpy as np

from detector import YOLODetector
from earphone_detector_local import LocalEarphoneDetector
from sender import send_event, upload_evidence
from utils import (
    current_timestamp,
    draw_bboxes,
    CooldownManager,
    EvidenceManager,
    VideoClipBuffer,
    ensure_dir,
)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("capture")

# Default suspicious classes to detect (can be overridden via CLI)
DEFAULT_CLASSES = [
    "person",
    "cell phone",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "remote",
    "backpack",
    "handbag",
    "tv",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to YOLO weights e.g. yolov8n.pt")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--backend", default=os.getenv("BACKEND_URL", "http://localhost:5000/api/event"))
    p.add_argument("--camera", default=0, help="Camera index or video path")
    p.add_argument("--dry-run", action="store_true", help="Do not send events to backend")
    p.add_argument("--save-video", action="store_true", help="Save short clip (clip-seconds) around event")
    p.add_argument("--clip-seconds", type=float, default=6.0, help="Seconds of clip to save around event")
    p.add_argument("--fps", type=int, default=20, help="Target FPS for capture loop")
    p.add_argument("--evidence-ttl", type=int, default=600, help="Evidence TTL (seconds) to cleanup")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="List of target class names")
    p.add_argument("--evidence-dir", default="evidence", help="Folder to save short-lived evidence")
    return p.parse_args()


def build_event(detections: List[dict], camera_id: str = "cam0", session_id: Optional[str] = None, evidence_image: Optional[str] = None) -> dict:
    """
    Build JSON event following the required schema.
    """
    ts = current_timestamp()
    max_conf = max((d["confidence"] for d in detections), default=0.0)
    person_count = sum(1 for d in detections if d["label"].lower() == "person")
    phone_detected = any(d["label"].lower() in ("cell phone", "mobile phone", "phone") for d in detections)
    earphone_detected = any("earphone" in d["label"].lower() for d in detections)

    summary = {
        "phone_detected": bool(phone_detected),
        "earphone_detected": bool(earphone_detected),
        "person_count": int(person_count),
        "max_confidence": float(max_conf),
    }

    event = {
        "module": "yolo_object_detection",
        "timestamp": float(ts),
        "session_id": session_id,
        "camera_id": camera_id,
        "detections": detections,
        "summary": summary,
        "evidence_image": os.path.basename(evidence_image) if evidence_image else None,
    }
    return event


def start_background_send(event_json: dict, evidence_path: Optional[str], backend_url: str, dry_run: bool = False):
    """
    Launch a background thread to send event (and possibly upload evidence).
    """

    def _worker():
        try:
            if dry_run:
                logger.info("[DRY-RUN] would send event: %s", event_json)
                return

            if evidence_path and os.path.exists(evidence_path):
                r = upload_evidence(event_json, evidence_path, backend_url)
                if r is None:
                    send_event(event_json, backend_url)
                else:
                    try:
                        os.remove(evidence_path)
                    except Exception:
                        logger.exception("Failed deleting evidence")
            else:
                send_event(event_json, backend_url)

        except Exception:
            logger.exception("Background send failed")

    th = threading.Thread(target=_worker, daemon=True)
    th.start()


def main():
    args = parse_args()
    ensure_dir(args.evidence_dir)

    # instantiate YOLO detector
    detector = YOLODetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        classes=args.classes,
        imgsz=320,
    )

    detector.suspicious_objects = set(
        name.lower() for name in args.classes if name.lower() in detector.DEFAULT_SUSPICIOUS
    ) | {"cell phone", "book", "laptop"}

    # Earphone detector
    try:
        earphone_detector = LocalEarphoneDetector(conf_threshold=0.25)
        logger.info("Earphone detector initialized (mAP: 89.4%, conf: 0.25)")
    except Exception as e:
        logger.error("Earphone detector failed: %s", e)
        earphone_detector = None

    # Managers
    cooldown = CooldownManager(default_cooldown=8.0)
    evidence_mgr = EvidenceManager(args.evidence_dir, ttl_seconds=args.evidence_ttl)
    clip_buffer = VideoClipBuffer(args.clip_seconds, args.fps) if args.save_video else None

    # Capture
    cam_src = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(cam_src)

    if not cap.isOpened():
        logger.error("Unable to open camera/source: %s", args.camera)
        sys.exit(1)

    running = True

    def _signal_handler(sig, frame):
        nonlocal running
        logger.info("SIGINT received, stopping...")
        running = False

    signal.signal(signal.SIGINT, _signal_handler)

    target_period = 1.0 / max(1, args.fps)
    last_cleanup = time.time()

    logger.info("Capture loop started...")
    logger.info("Camera window should appear. Press 'q' to quit.")

    try:
        while running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed.")
                continue

            if clip_buffer:
                clip_buffer.add(frame)

            detections, annotated = detector.detect(frame, annotate=True)

            # Earphone detection
            if earphone_detector:
                try:
                    ear_dets = earphone_detector.detect(frame)
                    if ear_dets:
                        detections.extend(ear_dets)
                        for ed in ear_dets:
                            x1, y1, x2, y2 = ed["bbox"]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(annotated, f"{ed['label']} {ed['confidence']:.2f}", 
                                      (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                except Exception:
                    pass

            # Check for earphones
            earphone_items = [d for d in detections if "earphone" in d["label"].lower()]
            if earphone_items and cooldown.can_fire("earphone_detected"):
                cooldown.fired("earphone_detected")
                ev_path = None
                try:
                    ev_path = evidence_mgr.save_jpg(annotated, prefix="earphone")
                except Exception:
                    pass
                event = build_event(detections, camera_id=str(args.camera), evidence_image=ev_path)
                start_background_send(event, ev_path, args.backend, args.dry_run)

            person_count = sum(1 for d in detections if d["label"].lower() == "person")
            phone_items = [d for d in detections if "phone" in d["label"].lower()]
            suspicious_items = [
                d for d in detections
                if d["label"].lower() in detector.suspicious_objects
            ]

            if person_count > 1 and cooldown.can_fire("multi_person"):
                cooldown.fired("multi_person")
                event = build_event(detections, camera_id=str(args.camera))
                start_background_send(event, None, args.backend, args.dry_run)

            if phone_items and cooldown.can_fire("phone_detected"):
                cooldown.fired("phone_detected")
                best = max(phone_items, key=lambda d: d["confidence"])
                ev_path = None
                if best["confidence"] >= args.conf:
                    try:
                        ev_path = evidence_mgr.save_jpg(annotated, prefix="phone")
                    except Exception:
                        pass

                event = build_event(detections, camera_id=str(args.camera), evidence_image=ev_path)
                start_background_send(event, ev_path, args.backend, args.dry_run)

            for item in suspicious_items:
                event_type = f"object_{item['label'].lower()}"
                if cooldown.can_fire(event_type):
                    cooldown.fired(event_type)
                    ev_path = None
                    if item["confidence"] >= args.conf:
                        try:
                            ev_path = evidence_mgr.save_jpg(
                                annotated, prefix=item["label"].replace(" ", "_")
                            )
                        except Exception:
                            pass

                    event = build_event([item], camera_id=str(args.camera), evidence_image=ev_path)
                    start_background_send(event, ev_path, args.backend, args.dry_run)

            # ----------------------------------------------

            cv2.imshow("YOLO Proctor (press q to quit)", annotated if annotated is not None else frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Cleanup expired evidence
            if time.time() - last_cleanup > 30:
                try:
                    evidence_mgr.cleanup()
                except Exception:
                    logger.exception("Cleanup failed.")
                last_cleanup = time.time()

            # FPS throttle
            elapsed = time.time() - t0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Capture loop ended.")


if __name__ == "__main__":
    main()
