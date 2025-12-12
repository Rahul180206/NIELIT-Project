import cv2
import numpy as np

try:
    import supervision as sv
    HAS_SUPERVISION = False  # force disable for performance
except Exception:
    HAS_SUPERVISION = False



def draw_boxes(frame, detections):
    """
    Draw YOLO detections using supervision (if available) or fallback to OpenCV.
    This implementation is tolerant of different supervision versions:
      - Tries to call BoxAnnotator.annotate(...) with labels.
      - If that fails (TypeError), falls back to annotate without labels
        and overlays labels manually using OpenCV.
    """

    # If no detections, return original frame
    if detections is None or len(detections) == 0:
        return frame

    if HAS_SUPERVISION:
        # Prepare arrays and labels
        xyxy = []
        confidences = []
        labels = []

        for d in detections:
            xyxy.append([d["x1"], d["y1"], d["x2"], d["y2"]])
            confidences.append(float(d["confidence"]))
            labels.append(f"{d['class_name']} {d['confidence']:.2f}")

        xyxy = np.array(xyxy, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        class_ids = np.zeros(len(xyxy), dtype=np.int32)

        # Create supervision Detections object (constructor works across versions)
        det = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)

        annotator = sv.BoxAnnotator()

        # Try annotate with labels first; fallback if API differs
        try:
            # Many supervision versions accept labels keyword
            output = annotator.annotate(scene=frame.copy(), detections=det, labels=labels)
            return output
        except TypeError:
            # Fallback: call annotate without labels then draw labels manually
            try:
                output = annotator.annotate(frame.copy(), det)
            except TypeError:
                # Last-resort: if annotate's signature is different, draw boxes manually
                output = frame.copy()
                for box in xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Overlay labels manually
            for (box, label) in zip(xyxy, labels):
                x1, y1, x2, y2 = map(int, box)
                text_pos = (x1, max(y1 - 6, 12))
                # background rectangle for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(output, (x1, text_pos[1] - th - 4), (x1 + tw + 4, text_pos[1] + 2), (0, 255, 0), -1)
                cv2.putText(output, label, (x1 + 2, text_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

            return output

    else:
        # OpenCV fallback: draw boxes and labels manually
        output = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            label = f"{d['class_name']} {d['confidence']:.2f}"

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        return output
