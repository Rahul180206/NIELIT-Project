import cv2
from src.detector import YOLODetector
from src.supervision_visualizer import draw_boxes

# Path to your test image (use any image you have)
IMAGE_PATH = "test.jpg"

def main():
    # Load model
    detector = YOLODetector(
        model_path="models/yolov8n.pt",
        device="cpu",
        conf=0.35,
        classes=['person','cellphone','laptop','book','tv']
    )

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ ERROR: Image not found at", IMAGE_PATH)
        return

    # Run detection
    detections = detector.predict_frame(img)
    print("Detections:", detections)

    # Draw boxes
    out = draw_boxes(img, detections)

    # Show output
    cv2.imshow("YOLO Test Image Detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
