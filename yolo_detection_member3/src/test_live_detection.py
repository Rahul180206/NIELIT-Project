import sys
import os
import cv2
import time
import threading
import argparse

# Add project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.detector import YOLODetector
from src.supervision_visualizer import draw_boxes
from src.evidence_capture import handle_event


# ============================
# EVENT RATE LIMIT SETUP
# ============================
last_event_time = {
    "multiple_persons": 0,
    "phone_detected": 0,
    "laptop_detected": 0,
    "notebook_detected": 0
}

EVENT_COOLDOWN = 3  # seconds


def event_trigger(event_name, frame, detections):
    """Triggers event asynchronously to avoid lag."""
    if time.time() - last_event_time[event_name] > EVENT_COOLDOWN:
        last_event_time[event_name] = time.time()

        print(f"⚠ EVENT: {event_name.replace('_', ' ').title()}")

        threading.Thread(
            target=handle_event,
            args=(frame.copy(), detections, event_name)
        ).start()


def main():

    # ---------------------------
    # Parse command-line arguments
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolov8n.pt",
                        help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_path = args.model

    classes_to_detect = ["person", "cell phone", "laptop", "tv", "book"]

    detector = YOLODetector(
        model_path=model_path,
        device=args.device,
        conf=0.20,
        iou=0.30,
        classes=classes_to_detect
    )

    cap = cv2.VideoCapture(0)

    # Set lower resolution for FPS boost
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam")
        return

    print("✔ Webcam started... Press Q to quit.")
    print(f"✔ Running model: {model_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error")
            break

        detections = detector.predict_frame(frame)

        # Split detections by type
        persons = [d for d in detections if d["class_name"] == "person"]
        phones = [d for d in detections if d["class_name"] == "cell phone"]
        laptops = [d for d in detections if d["class_name"] == "laptop"]
        notebooks = [d for d in detections if d["class_name"] == "book"]

        # ---------------------------
        # EVENT LOGIC
        # ---------------------------
        if len(persons) > 1:
            event_trigger("multiple_persons", frame, detections)

        if len(phones) > 0:
            event_trigger("phone_detected", frame, detections)

        if len(laptops) > 0:
            event_trigger("laptop_detected", frame, detections)

        if len(notebooks) > 0:
            event_trigger("notebook_detected", frame, detections)

        # Draw boxes after event triggers
        output = draw_boxes(frame, detections)
        cv2.imshow("Live Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
