# src/train_earphone.py
import os
import argparse
import shutil
from ultralytics import YOLO

def train(dataset_root, model='yolov8n.pt', epochs=30, batch=16, imgsz=640, name='earphone_exp'):

    # Always use absolute yaml path (no YOLO repetition bug)
    data_yaml = os.path.abspath("datasets/earphone/data.yaml")
    print("[INFO] Using data yaml:", data_yaml)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError("data.yaml NOT FOUND at datasets/earphone/data.yaml")

    # Load YOLO model
    model = YOLO(model)
    print("[INFO] Training started...\n")

    # Train YOLO
    model.train(
        data=data_yaml,     # IMPORTANT: absolute path only
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name
    )

    # YOLO 8.3+ stores detect models inside: runs/detect/{exp_name}/weights/best.pt
    origin = f"C:/Users/rahul/runs/detect/{name}/weights/best.pt"

    if os.path.exists(origin):
        os.makedirs("models", exist_ok=True)
        destination = "models/earphone_best.pt"

        shutil.copyfile(origin, destination)
        print(f"[INFO] Best model saved to: {destination}")
    else:
        print("[WARN] best.pt was NOT found. Training may have failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/earphone")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train(
        dataset_root=args.dataset,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
