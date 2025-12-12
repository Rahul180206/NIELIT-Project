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
import os
from datetime import datetime

# COCO class IDs for our target classes
COCO_CLASSES = {
    0: 'person',
    67: 'cell phone', 
    63: 'laptop',
    73: 'book',
    62: 'tv'
}

def detect_combined(earphone_model_path, source, show=True, save=False, conf=0.3):
    # Create evidence directory if it doesn't exist
    evidence_dir = "../evidence"
    os.makedirs(evidence_dir, exist_ok=True)
    print(f"[INFO] Evidence will be saved to: {os.path.abspath(evidence_dir)}")
    print("[INFO] Loading models...")
    
    try:
        # Load pre-trained YOLO for COCO classes
        coco_model = YOLO('yolov8n.pt')
        print("[INFO] COCO model loaded successfully")
        
        # Load custom earphone model with error handling
        if not os.path.exists(earphone_model_path):
            print(f"[ERROR] Earphone model not found at: {earphone_model_path}")
            return
        
        earphone_model = YOLO(earphone_model_path)
        print(f"[INFO] Earphone model loaded from: {earphone_model_path}")
        
        # Test earphone model
        print(f"[INFO] Earphone model classes: {earphone_model.names}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return
    
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
            
            # Detect earphones with very low confidence
            earphone_conf = 0.05  # Very low confidence
            earphone_results = earphone_model.predict(frame, conf=earphone_conf, verbose=False)
            
            # Process and display results
            annotated_frame = frame.copy()
            detections_found = False
            person_count = 0
            prohibited_items = []
            
            # Process COCO detections
            for r in coco_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in COCO_CLASSES:
                            detections_found = True
                            conf_score = float(box.conf[0])
                            class_name = COCO_CLASSES[cls_id]
                            
                            # Count persons and track prohibited items
                            if class_name == 'person':
                                person_count += 1
                            elif class_name in ['cell phone', 'laptop', 'tv']:
                                prohibited_items.append(class_name)
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{class_name}: {conf_score:.2f}"
                            cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            print(f"[DETECTION] {class_name}: {conf_score:.2f}")
            
            # Process earphone detections - show all
            for r in earphone_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        detections_found = True
                        conf_score = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Draw bounding box (blue for earphones)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        
                        # Draw label
                        label = f"earphone: {conf_score:.2f}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        print(f"[DETECTION] earphone: {conf_score:.2f}")
                        prohibited_items.append('earphone')
            
            # Save evidence for violations
            if save and (len(prohibited_items) > 0 or person_count > 1):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if person_count > 1:
                    filename = f"multiple_persons_{timestamp}_{hash(str(timestamp)) % 100000:06d}.jpg"
                    filepath = os.path.join(evidence_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"[EVIDENCE] Multiple persons detected - Saved: {filepath}")
                
                for item in set(prohibited_items):  # Remove duplicates
                    filename = f"{item.replace(' ', '_')}_detected_{timestamp}_{hash(str(item+timestamp)) % 100000:06d}.jpg"
                    filepath = os.path.join(evidence_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"[EVIDENCE] {item} detected - Saved: {filepath}")
            
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
        
        # Detect earphones with very low confidence
        earphone_conf = 0.05  # Very low confidence
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
        
        # Process earphone detections - show all
        for r in earphone_results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    detections_found = True
                    conf_score = float(box.conf[0])
                    print(f"[DETECTION] earphone: {conf_score:.2f}")
        
        if not detections_found:
            print("[INFO] No target objects detected")
    
    print("[INFO] Detection finished.")

def test_earphone_model_only(model_path):
    """Test earphone model independently"""
    print(f"[TEST] Testing earphone model: {model_path}")
    
    try:
        model = YOLO(model_path)
        print(f"[TEST] Model loaded. Classes: {model.names}")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        
        print("[TEST] Press 'q' to quit test")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            results = model.predict(frame, conf=0.05, verbose=False)  # Very low confidence for testing
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    print(f"[TEST FRAME {frame_count}] Earphone detected!")
                    for box in r.boxes:
                        conf_score = float(box.conf[0])
                        print(f"  Confidence: {conf_score:.3f}")
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"earphone: {conf_score:.2f}", (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Earphone Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--earphone-model", type=str, default="models/earphone_best.pt", 
                       help="Path to earphone model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or file path")
    parser.add_argument("--show", action="store_true", help="Display output window")
    parser.add_argument("--save", action="store_true", help="Save evidence images to evidence folder")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--test-earphone", action="store_true", help="Test earphone model only")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.test_earphone:
        test_earphone_model_only(args.earphone_model)
    else:
        source = 0 if args.source == "0" else args.source
        detect_combined(
            earphone_model_path=args.earphone_model,
            source=source,
            show=args.show,
            save=args.save,
            conf=args.conf
        )
