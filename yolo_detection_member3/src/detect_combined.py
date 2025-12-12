#!/usr/bin/env python3
"""
Combined detection script that uses:
1. Pre-trained YOLOv8 for COCO classes (person, cell phone, laptop, book, tv)
2. Custom trained model for earphone detection
"""

import cv2
import argparse
from ultralytics import YOLO
import numpy as np

# COCO class IDs for our target classes
COCO_CLASSES = {
    0: 'person',
    67: 'cell phone', 
    63: 'laptop',
    73: 'book',
    62: 'tv'
}

def detect_combined(earphone_model_path, source, show=True, save=False, conf=0.3):
    print("[INFO] Loading models...")
    
    # Load pre-trained YOLO for COCO classes
    coco_model = YOLO('yolov8n.pt')
    
    # Load custom earphone model
    earphone_model = YOLO(earphone_model_path)
    
    print("[INFO] Models loaded successfully")
    print(f"[INFO] Target classes: {list(COCO_CLASSES.values())} + earphone")
    print(f"[INFO] Confidence threshold: {conf}")
    
    if source == 0:
        # Webcam detection
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        
        print("[INFO] Starting webcam detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect COCO classes
            coco_results = coco_model.predict(frame, conf=conf, verbose=False)
            
            # Detect earphones with very low confidence threshold
            earphone_conf = 0.1  # Very low confidence for earphones
            earphone_results = earphone_model.predict(frame, conf=earphone_conf, verbose=False)
            
            # Process and display results
            annotated_frame = frame.copy()
            detections_found = False
            
            # Process COCO detections
            for r in coco_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in COCO_CLASSES:
                            detections_found = True
                            conf_score = float(box.conf[0])
                            class_name = COCO_CLASSES[cls_id]
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{class_name}: {conf_score:.2f}"
                            cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            print(f"[DETECTION] {class_name}: {conf_score:.2f}")
            
            # Process earphone detections
            for r in earphone_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        detections_found = True
                        conf_score = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Draw bounding box (different color for earphones)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        
                        # Draw label
                        label = f"earphone: {conf_score:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        print(f"[DETECTION] earphone: {conf_score:.2f}")
            
            if not detections_found:
                print("[INFO] No target objects detected in current frame")
            
            if show:
                cv2.imshow('Multi-Class Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # Single image/video detection
        print(f"[INFO] Processing: {source}")
        
        # Detect COCO classes
        coco_results = coco_model.predict(source, conf=conf, verbose=False)
        
        # Detect earphones with very low confidence threshold
        earphone_conf = 0.1  # Very low confidence for earphones
        earphone_results = earphone_model.predict(source, conf=earphone_conf, verbose=False)
        
        detections_found = False
        
        # Process COCO detections
        for r in coco_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in COCO_CLASSES:
                        detections_found = True
                        conf_score = float(box.conf[0])
                        class_name = COCO_CLASSES[cls_id]
                        print(f"[DETECTION] {class_name}: {conf_score:.2f}")
        
        # Process earphone detections
        for r in earphone_results:
            if r.boxes is not None:
                for box in r.boxes:
                    detections_found = True
                    conf_score = float(box.conf[0])
                    print(f"[DETECTION] earphone: {conf_score:.2f}")
        
        if not detections_found:
            print("[INFO] No target objects detected")
    
    print("[INFO] Detection finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--earphone-model", type=str, default="models/earphone_best.pt", 
                       help="Path to earphone model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or file path")
    parser.add_argument("--show", action="store_true", help="Display output window")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source

    detect_combined(
        earphone_model_path=args.earphone_model,
        source=source,
        show=args.show,
        save=args.save,
        conf=args.conf
    )