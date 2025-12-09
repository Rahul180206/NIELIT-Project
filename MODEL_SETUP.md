# Model Setup Guide

## Earphone Detection Model

The earphone detection model is too large to include in the GitHub repository. Follow these steps to set it up:

### Option 1: Download Pre-trained Model

1. Download the trained model from: **[Add your Google Drive/Dropbox link here]**
2. Extract and place `best.pt` in the following location:
   ```
   C:\Users\YOUR_USERNAME\runs\detect\earphone_worn_detector2\weights\best.pt
   ```
3. Update the path in `earphone_detector_local.py` if needed

### Option 2: Train Your Own Model

1. Collect 50-100 images of people wearing earphones
2. Annotate them using [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg)
3. Export in YOLOv8 format
4. Update `data.yaml` with your dataset paths
5. Run training:
   ```bash
   python train_earphone_new.py
   ```
6. Model will be saved in `C:\Users\YOUR_USERNAME\runs\detect\earphone_worn_detector2\weights\best.pt`

### Model Performance

- **mAP50**: 89.4%
- **Precision**: 80.1%
- **Recall**: 88.9%
- **Training Time**: ~25 minutes on CPU
- **Model Size**: 6.2 MB

### YOLO Base Model

The YOLOv8n base model (`yolov8n.pt`) will be automatically downloaded on first run.

### Updating Model Path

If your model is in a different location, update line 11 in `earphone_detector_local.py`:

```python
def __init__(self, model_path: str = r"YOUR_PATH_HERE\best.pt", conf_threshold: float = 0.25):
```

## Troubleshooting

**Model not found error:**
- Check the path in `earphone_detector_local.py`
- Ensure `best.pt` exists in the specified location
- Use absolute path with raw string (r"C:\...")

**Low accuracy:**
- Retrain with more diverse images
- Increase training epochs
- Use better lighting conditions during capture
