@echo off
echo Starting Multi-Class Object Detection...
echo Detecting: person, cell phone, laptop, book, tv, earphone
echo.
echo Press Ctrl+C to stop detection
echo Press 'q' in the video window to quit
echo.
python src/detect_combined.py --show --conf 0.3
pause