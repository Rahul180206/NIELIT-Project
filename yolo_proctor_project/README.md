# YOLO Proctor - AI-Powered Exam Proctoring System

Real-time object detection system for exam proctoring using YOLOv8 and custom-trained models.

## Features

- 🎯 **Multi-Object Detection**: Person, cell phone, laptop, notebook, earphones
- 📸 **Real-time Monitoring**: Live camera feed with bounding boxes
- 🔔 **Event System**: Cooldown-based alerts to prevent spam
- 💾 **Evidence Collection**: Automatic screenshot capture with TTL cleanup
- 🚀 **High Performance**: Optimized for real-time detection (20 FPS)
- 🎧 **Custom Earphone Detection**: Trained model with 89.4% mAP

## Detections

| Object | Status | Color | Accuracy |
|--------|--------|-------|----------|
| Person | ✅ | Green | High |
| Cell Phone | ✅ | Red | High |
| Laptop | ✅ | Green | High |
| Book/Notebook | ✅ | Green | High |
| Earphones | ✅ | Purple | 89.4% mAP |
| Keyboard/Mouse | ✅ | Green | High |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/yolo_proctor.git
cd yolo_proctor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model:
```bash
# YOLOv8 nano model will be downloaded automatically on first run
```

4. Download the trained earphone model:
- Download from: [Google Drive Link - Add your link here]
- Place `best.pt` in: `C:\Users\YOUR_USERNAME\runs\detect\earphone_worn_detector2\weights\`
- Or update the path in `earphone_detector_local.py`

## Usage

### Basic Usage (Dry Run)
```bash
python capture.py --model yolov8n.pt --dry-run
```

### With Backend Integration
```bash
python capture.py --model yolov8n.pt --backend http://localhost:5000/api/event
```

### Custom Configuration
```bash
python capture.py --model yolov8n.pt --fps 20 --conf 0.5 --dry-run
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Required | Path to YOLO weights (e.g., yolov8n.pt) |
| `--device` | auto | Device: auto, cpu, or cuda |
| `--conf` | 0.5 | Confidence threshold (0-1) |
| `--backend` | localhost:5000 | Backend API URL |
| `--camera` | 0 | Camera index or video path |
| `--dry-run` | False | Test mode (no backend calls) |
| `--fps` | 20 | Target FPS |
| `--evidence-dir` | evidence | Evidence storage folder |

## Project Structure

```
yolo_proctor/
├── capture.py                    # Main detection loop
├── detector.py                   # YOLO detector wrapper
├── earphone_detector_local.py    # Custom earphone detector
├── sender.py                     # Backend communication
├── utils.py                      # Helper utilities
├── train_earphone_new.py         # Training script
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── evidence/                     # Evidence images (auto-cleanup)
```

## Training Custom Earphone Model

1. Prepare your dataset in YOLO format
2. Update `data.yaml` with correct paths
3. Run training:
```bash
python train_earphone_new.py
```

## Configuration

Create a `.env` file (optional):
```env
BACKEND_URL=http://localhost:5000/api/event
```

## System Requirements

- Python 3.8+
- Webcam or video source
- CPU (works) or GPU (recommended for better performance)
- 4GB RAM minimum

## Performance

- **FPS**: 15-20 on CPU, 30+ on GPU
- **Detection Latency**: <100ms
- **Model Size**: ~6MB (YOLOv8n + custom earphone model)

## Event System

The system uses cooldown-based event triggering:
- **Multi-person**: 8 second cooldown
- **Phone detected**: 8 second cooldown
- **Earphone detected**: 8 second cooldown
- **Other objects**: 8 second cooldown per object type

## Evidence Management

- Evidence images saved with timestamps
- Automatic cleanup after 10 minutes (configurable)
- Organized by detection type (phone_, earphone_, laptop_, etc.)

## API Integration

Events are sent to backend with the following schema:
```json
{
  "module": "yolo_object_detection",
  "timestamp": 1234567890.123,
  "camera_id": "0",
  "detections": [...],
  "summary": {
    "phone_detected": true,
    "earphone_detected": true,
    "person_count": 1,
    "max_confidence": 0.95
  },
  "evidence_image": "earphone_1234567890.jpg"
}
```

## Troubleshooting

**Camera not opening:**
- Check camera permissions
- Try different camera index: `--camera 1`

**Low FPS:**
- Reduce image size in detector.py (imgsz=320)
- Use GPU if available: `--device cuda`

**Earphones not detected:**
- Ensure good lighting
- Earphones should be clearly visible
- Lower confidence threshold in `earphone_detector_local.py`

## License

This project is licensed under the MIT License.

## Acknowledgments

- YOLOv8 by Ultralytics
- Training dataset created with Roboflow
- Custom earphone detection model trained on personal dataset

## Contact

For questions or issues, please open an issue on GitHub.
