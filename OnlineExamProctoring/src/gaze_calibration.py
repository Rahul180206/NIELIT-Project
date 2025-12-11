import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris landmarks
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def get_iris_center(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices])
    c = pts.mean(axis=0)
    return (int(c[0]), int(c[1]))

def normalize(point, box):
    left, top, right, bottom = box
    w = max(1, right - left)
    h = max(1, bottom - top)
    nx = (point[0] - left) / w
    ny = (point[1] - top) / h
    return nx, ny

def eye_box(landmarks, eye_idx):
    xs = [landmarks[i][0] for i in eye_idx]
    ys = [landmarks[i][1] for i in eye_idx]
    return (min(xs), min(ys), max(xs), max(ys))

def show_instruction(frame, text):
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)


# ======================================================
# CALIBRATION PROCESS
# ======================================================
def record_direction(direction_name, duration_seconds, face_mesh):
    print(f"\n[CALIB] LOOK {direction_name} for {duration_seconds} seconds...\n")
    time.sleep(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    samples = []

    start = time.time()

    while time.time() - start < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # convert to pixel landmarks
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

            eye_L = get_iris_center(pts, LEFT_IRIS)
            eye_R = get_iris_center(pts, RIGHT_IRIS)

            box_L = eye_box(pts, LEFT_EYE)
            box_R = eye_box(pts, RIGHT_EYE)

            nxL, nyL = normalize(eye_L, box_L)
            nxR, nyR = normalize(eye_R, box_R)

            nx = (nxL + nxR) / 2
            ny = (nyL + nyR) / 2

            samples.append((nx, ny))

            show_instruction(frame, f"LOOK {direction_name}")
        else:
            show_instruction(frame, f"NO FACE DETECTED")

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return samples


def main():
    print("\n======================================")
    print("     ADVANCED GAZE CALIBRATION")
    print("======================================\n")
    print("Follow the instructions on-screen.\n")

    os.makedirs("config", exist_ok=True)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        # RECORD CENTER FIRST
        center_samples = record_direction("CENTER", 5, fm)

        # LEFT
        left_samples = record_direction("LEFT", 3, fm)

        # RIGHT
        right_samples = record_direction("RIGHT", 3, fm)

        # UP
        up_samples = record_direction("UP", 3, fm)

        # DOWN
        down_samples = record_direction("DOWN", 3, fm)

    # Compute averages
    def avg(samples, axis):   # axis 0 = nx, axis 1 = ny
        return float(np.mean([s[axis] for s in samples]))

    config = {
        "center_nx": avg(center_samples, 0),
        "center_ny": avg(center_samples, 1),

        "left_nx": avg(left_samples, 0),
        "right_nx": avg(right_samples, 0),

        "up_ny": avg(up_samples, 1),
        "down_ny": avg(down_samples, 1),

        # automatic thresholds
        "left_thresh": (avg(center_samples, 0) + avg(left_samples, 0)) / 2,
        "right_thresh": (avg(center_samples, 0) + avg(right_samples, 0)) / 2,

        "up_thresh": (avg(center_samples, 1) + avg(up_samples, 1)) / 2,
        "down_thresh": (avg(center_samples, 1) + avg(down_samples, 1)) / 2
    }

    # Save config
    with open("config/gaze_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n======================================")
    print("  CALIBRATION COMPLETE")
    print("  Saved to config/gaze_config.json")
    print("======================================\n")


if __name__ == "__main__":
    main()
