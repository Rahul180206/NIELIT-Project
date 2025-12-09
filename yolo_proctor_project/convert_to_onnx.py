"""
convert_to_onnx.py
Convert ultralytics .pt to .onnx (uses ultralytics export).
Example:
    python convert_to_onnx.py --weights yolov8n.pt --out yolov8n.onnx
"""

import argparse
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.weights)
    logging.info("Exporting to ONNX...")
    model.export(format="onnx", imgsz=args.imgsz, overwrite=True)
    logging.info("Export complete. Check model folder for .onnx file.")


if __name__ == "__main__":
    main()
