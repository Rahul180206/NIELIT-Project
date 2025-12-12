#!/usr/bin/env python3
"""
Test earphone detection with webcam
"""

import cv2
from ultralytics import YOLO

def test_earphone_webcam():
    print("Testing earphone detection with webcam...")
    
    # Load earphone model
    model = YOLO('models/earphone_best.pt')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection with very low confidence
        results = model.predict(frame, conf=0.1, verbose=False)
        
        # Check for detections
        detections_found = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                detections_found = True
                for box in r.boxes:
                    conf_score = float(box.conf[0])
                    print(f"[FRAME {frame_count}] EARPHONE DETECTED: {conf_score:.3f}")
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"earphone: {conf_score:.2f}", (int(x1), int(y1-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if not detections_found and frame_count % 30 == 0:  # Print every 30 frames
            print(f"[FRAME {frame_count}] No earphones detected")
        
        # Show frame
        cv2.imshow('Earphone Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_earphone_webcam()