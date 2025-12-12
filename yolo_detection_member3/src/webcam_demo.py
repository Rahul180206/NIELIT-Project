# src/webcam_demo.py
import cv2
import argparse
from src.detector import YOLODetector
from src.supervision_visualizer import draw_boxes
from src.evidence_capture import handle_event
import time
from collections import deque

SUSPICIOUS_PHONE = True
SUSPICIOUS_MULTIPLE_PERSON = True
PERSON_THRESHOLD = 1  # more than this => suspicious
FRAME_SAVE_INTERVAL = 2  # seconds between saving evidence for same event

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='models/yolov8n.pt')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda:0')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--source', type=int, default=0)
    p.add_argument('--display', action='store_true', default=True)
    p.add_argument('--save-video', action='store_true', default=False)
    return p.parse_args()

def main():
    args = parse_args()
    detector = YOLODetector(model_path=args.model, device=args.device, conf=args.conf,
                            classes=['person','cellphone','laptop','book','tv'])
    cap = cv2.VideoCapture(args.source)
    out_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter('output_record.mp4', fourcc, 20.0,
                                     (int(cap.get(3)), int(cap.get(4))))
    last_saved = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detections = detector.predict_frame(frame)
        # simple rule checks
        persons = [d for d in detections if d['class_name'] == 'person']
        phones = [d for d in detections if 'phone' in d['class_name'] or 'cell' in d['class_name']]
        laptops = [d for d in detections if d['class_name'] == 'laptop']
        books = [d for d in detections if d['class_name'] == 'book' or d['class_name']=='notebook']
        tvs = [d for d in detections if d['class_name'] == 'tv']

        # build nice visual
        vis = draw_boxes(frame, detections)

        event = None
        now = time.time()

        if SUSPICIOUS_PHONE and len(phones) > 0:
            event = 'phone_detected'
        elif SUSPICIOUS_MULTIPLE_PERSON and len(persons) > PERSON_THRESHOLD:
            event = 'multiple_persons'
        # (extend rules as needed)
        if event:
            # rate-limit same event to avoid flooding
            last = last_saved.get(event, 0)
            if now - last > FRAME_SAVE_INTERVAL:
                img_path, json_path = handle_event(frame, detections, event_type=event, frame_idx=frame_idx)
                print(f"[EVENT] {event} saved: {img_path}, {json_path}")
                last_saved[event] = now

        if args.display:
            cv2.imshow("YOLO Detection", vis)
            if out_writer:
                out_writer.write(vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
