# src/detect_multi.py

import cv2
import argparse
from ultralytics import YOLO
import yaml
import os

def load_class_names(data_yaml_path):
    """Load class names from data.yaml file"""
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    return []

def detect(model_path, source, show=True, save=False, conf=0.3):
    print("[INFO] Loading model:", model_path)
    model = YOLO(model_path)
    
    # Load class names from multi dataset config
    data_yaml_path = "datasets/multi/data.yaml"
    class_names = load_class_names(data_yaml_path)
    
    if class_names:
        print(f"[INFO] Loaded {len(class_names)} classes: {class_names}")
    else:
        print("[WARNING] Could not load class names from data.yaml")
    
    print(f"[INFO] Starting detection with confidence threshold: {conf}")
    
    results = model.predict(
        source=source,
        show=show,
        save=save,
        conf=conf,
        verbose=True
    )
    
    # Print detection results
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                if cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                print(f"[DETECTION] {class_name}: {conf_score:.2f}")
        else:
            print("[INFO] No objects detected in current frame")

    print("[INFO] Detection finished.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or file path")
    parser.add_argument("--show", action="store_true", help="Display output window")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source

    detect(
        model_path=args.model,
        source=source,
        show=args.show,
        save=args.save,
        conf=args.conf
    )