import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

def eye_box(pts, idx):
    xs = [pts[i][0] for i in idx]
    ys = [pts[i][1] for i in idx]
    return (min(xs), min(ys), max(xs), max(ys))

def iris_center(pts, idx):
    pts = np.array([pts[i] for i in idx], dtype=np.float32)
    c = pts.mean(axis=0)
    return (float(c[0]), float(c[1]))

def normalize(p, box):
    l, t, r, b = box
    w = max(1, r - l)
    h = max(1, b - t)
    return (p[0] - l) / w, (p[1] - t) / h

def capture_pose(direction, secs, fm):
    print(f"\nLOOK {direction} for {secs} seconds...")
    time.sleep(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    samples = []
    start = time.time()

    while time.time() - start < secs:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            pts = [(int(l.x*w), int(l.y*h)) for l in face.landmark]

            boxL = eye_box(pts, LEFT_EYE)
            boxR = eye_box(pts, RIGHT_EYE)

            irisL = iris_center(pts, LEFT_IRIS)
            irisR = iris_center(pts, RIGHT_IRIS)

            nxL, nyL = normalize(irisL, boxL)
            nxR, nyR = normalize(irisR, boxR)

            samples.append(((nxL+nxR)/2, (nyL+nyR)/2))

            cv2.putText(frame, f"LOOK {direction}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Calibration v2", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return samples

def avg(samples):
    return float(np.mean([s[0] for s in samples])), float(np.mean([s[1] for s in samples]))

def main():
    os.makedirs("config", exist_ok=True)

    print("CALIBRATION v2 STARTING")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as fm:

        center = capture_pose("CENTER", 5, fm)
        left   = capture_pose("LEFT",   3, fm)
        right  = capture_pose("RIGHT",  3, fm)
        up     = capture_pose("UP",     3, fm)

    cx, cy = avg(center)
    lx, ly = avg(left)
    rx, ry = avg(right)
    ux, uy = avg(up)

    config = {
        "center_nx": cx, "center_ny": cy,
        "left_nx": lx,   "right_nx": rx,
        "up_ny": uy,
        "left_thresh": (cx + lx)/2,
        "right_thresh": (cx + rx)/2,
        "up_thresh": (cy + uy)/2
    }

    with open("config/gaze_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nCALIBRATION COMPLETE!")
    print("Saved as config/gaze_config.json\n")

if __name__ == "__main__":
    main()
