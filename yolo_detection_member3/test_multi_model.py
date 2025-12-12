#!/usr/bin/env python3
"""
Quick test script to verify multi-class model functionality
"""

from ultralytics import YOLO
import yaml

def test_model():
    print("Testing multi-class model...")
    
    # Load the multi-class model
    model_path = "models/multi_best.pt"
    model = YOLO(model_path)
    
    # Load class names
    with open("datasets/multi/data.yaml", 'r') as f:
        data = yaml.safe_load(f)
        class_names = data.get('names', [])
    
    print(f"Model loaded: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Test with a sample image
    test_images = [
        "test/images/WIN_20251209_09_55_45_Pro_jpg.rf.863c6faec2fe99f4773ea3a5a7e2abc3.jpg",
        "test/images/WIN_20251209_09_56_14_Pro_jpg.rf.2be6edde52fff7a056a8753d06f0d1b4.jpg"
    ]
    
    for img_path in test_images:
        print(f"\nTesting with image: {img_path}")
        try:
            results = model.predict(img_path, conf=0.3, verbose=False)
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    print(f"Found {len(r.boxes)} detections:")
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        if cls_id < len(class_names):
                            class_name = class_names[cls_id]
                        else:
                            class_name = f"class_{cls_id}"
                        print(f"  - {class_name}: {conf_score:.3f}")
                else:
                    print("  No objects detected")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_model()