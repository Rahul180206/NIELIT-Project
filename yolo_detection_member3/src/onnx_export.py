# src/onnx_export.py
from ultralytics import YOLO
import argparse
import os

def export_to_onnx(model_path='models/yolov8n.pt', out_path='models/yolov8n.onnx', opset=13, dynamic=True):
    model = YOLO(model_path)
    print("Exporting to ONNX...")
    model.export(format='onnx', imgsz=640, opset=opset, dynamic=dynamic, device='cpu')
    # ultralytics will save in models/ automatically, show location
    print("Export done. Check models/ for exported onnx file.")

if __name__ == '__main__':
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/yolov8n.pt')
    p.add_argument('--out', default='models/yolov8n.onnx')
    p.add_argument('--opset', type=int, default=13)
    args = p.parse_args()
    export_to_onnx(args.model, args.out, args.opset)
