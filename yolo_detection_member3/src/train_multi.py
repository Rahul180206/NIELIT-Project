# src/train_multi.py

import os
import argparse
from ultralytics import YOLO
import shutil

def train(model_path="yolov8n.pt", epochs=50, batch=16, imgsz=640, name="multi_object_exp"):
    
    # absolute dataset yaml path
    data_yaml = os.path.abspath("datasets/multi/data.yaml")
    print("\n[INFO] Training using dataset:", data_yaml)

    model = YOLO(model_path)

    print("[INFO] Training started...\n")

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name
    )

    # YOLO 8.3.* stores runs under runs/detect/
    weights_src = f"C:/Users/rahul/runs/detect/{name}/weights/best.pt"

    if os.path.exists(weights_src):
        os.makedirs("models", exist_ok=True)
        dst = "models/multi_best.pt"
        shutil.copy(weights_src, dst)
        print("[INFO] Saved best trained model to:", dst)
    else:
        print("[WARN] best.pt NOT FOUND. Training likely failed or early stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    train(
        model_path=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
