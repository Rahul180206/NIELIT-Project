# src/accuracy_test.py
from ultralytics import YOLO
import argparse

def run_val(model_path, data_yaml):
    y = YOLO(model_path)
    print("Running validation (this computes metrics)...")
    res = y.val(data=data_yaml)
    print(res)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    args = p.parse_args()
    run_val(args.model, args.data)
