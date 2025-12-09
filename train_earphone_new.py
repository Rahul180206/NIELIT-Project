"""
Train NEW earphone detection model with images of earphones being worn
"""
from ultralytics import YOLO

# Load a pretrained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='D:/earphone new dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='earphone_worn_detector',
    patience=20,
    device='cpu'
)

print("Training complete!")
print(f"Best model saved at: runs/detect/earphone_worn_detector/weights/best.pt")
