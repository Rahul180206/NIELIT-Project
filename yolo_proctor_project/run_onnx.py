"""
run_onnx.py
Run inference using ONNX Runtime for CPU acceleration (alternative to ultralytics runtime).
This is a minimal wrapper demonstrating inference on a single frame.

Usage:
    python run_onnx.py --onnx yolov8n.onnx --image test.jpg
"""

import argparse
import numpy as np
import onnxruntime as ort
import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)


def preprocess(img, size=640):
    h, w = img.shape[:2]
    r = size / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w = size - new_w
    pad_h = size - new_h
    top, bottom = 0, pad_h
    left, right = 0, pad_w
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_trans = img_norm.transpose(2, 0, 1)[None, ...]
    return img_trans, r, (left, top)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=640)
    args = p.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    img = cv2.imread(args.image)
    inp, scale, pad = preprocess(img, size=args.size)
    start = time.time()
    outputs = sess.run(None, {sess.get_inputs()[0].name: inp})
    logging.info("ONNX outputs received. time=%.3f", time.time() - start)
    print("Outputs shape(s):", [o.shape for o in outputs])


if __name__ == "__main__":
    main()
