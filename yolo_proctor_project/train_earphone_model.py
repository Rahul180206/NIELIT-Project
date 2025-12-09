"""
Train earphone detection model using the downloaded dataset
"""
from ultralytics import YOLO

# Load a pretrained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='D:/earphones.v2i.yolov8/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='earphone_detector',
    patience=10,
    device='cpu'
)

print("Training complete!")
print(f"Best model saved at: runs/detect/earphone_detector/weights/best.pt")
