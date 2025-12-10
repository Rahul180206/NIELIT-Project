# gaze_tracking_offset_calibrated.py
# Final version calibrated using your real offsets
# - Removes DOWN (not possible with your webcam angle)
# - Uses your exact UP, LEFT, RIGHT thresholds
# - Provides stable CENTER classification

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque

# -------------------------
# Tiny Kalman Smoother
# -------------------------
class TinyKalman:
    def __init__(self, x0=0.0, P0=1.0, Q=1e-5, R=1e-2):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def update(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x


# FaceMesh indices
LEFT_EYE_IDX  = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
LEFT_IRIS     = [468,469,470,471]
RIGHT_IRIS    = [473,474,475,476]

# Output folders
os.makedirs("logs/evidence", exist_ok=True)
LOG_PATH = "logs/gaze_offset_events.jsonl"


# -----------------------------
# Helper functions
# -----------------------------
def eye_bbox(pts, idx):
    xs = [pts[i][0] for i in idx]
    ys = [pts[i][1] for i in idx]
    return min(xs), min(ys), max(xs), max(ys)

def center_of_points(pts, idx):
    arr = np.array([pts[i] for i in idx], dtype=np.float32)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1])

def iris_center_try(pts, idx):
    try:
        arr = np.array([pts[i] for i in idx], dtype=np.float32)
        c = arr.mean(axis=0)
        return float(c[0]), float(c[1])
    except:
        return None

def save_event(ev):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(ev) + "\n")
    print("[GAZE EVENT]", ev)


# ======================================
# Main Calibrated Gaze Tracker
# ======================================
class IrisOffsetGaze:
    def __init__(self):
        # Calibration values from YOUR REAL DATA
        self.UP_Y_THRESH = -0.28      # you go above this when looking up
        self.LEFT_X_THRESH = -0.06    # you cross this when looking left
        self.RIGHT_X_THRESH = 0.06    # you cross this when looking right

        self.kx = TinyKalman()
        self.ky = TinyKalman()
        self.alpha = 0.45

        self.prev_x = None
        self.prev_y = None

        self.sustained_frames = 10
        self.counters = {
            "LEFT": 0,
            "RIGHT": 0,
            "UP": 0,
            "CENTER": 0,
            "AWAY": 0
        }

        self.history = deque(maxlen=15)

    def smooth(self, x, y):
        sx = self.kx.update(x)
        sy = self.ky.update(y)
        if self.prev_x is None:
            ox, oy = sx, sy
        else:
            ox = self.alpha * sx + (1 - self.alpha) * self.prev_x
            oy = self.alpha * sy + (1 - self.alpha) * self.prev_y

        self.prev_x, self.prev_y = ox, oy
        return ox, oy

    # -----------------------------
    # Final calibrated classification
    # -----------------------------
    def classify_offset(self, x, y):
        if y < self.UP_Y_THRESH:
            return "UP"

        if x < self.LEFT_X_THRESH:
            return "LEFT"
        if x > self.RIGHT_X_THRESH:
            return "RIGHT"

        return "CENTER"

    # -----------------------------
    # Run
    # -----------------------------
    def run(self, cam_index=0):
        mpfm = mp.solutions.face_mesh
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        with mpfm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as fm:

            frame_no = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                gaze = "NO_FACE"

                if res.multi_face_landmarks:
                    face = res.multi_face_landmarks[0]
                    pts = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

                    # Eye centers
                    left_e_cx, left_e_cy = center_of_points(pts, LEFT_EYE_IDX)
                    right_e_cx, right_e_cy = center_of_points(pts, RIGHT_EYE_IDX)

                    # Iris centers
                    l_iris = iris_center_try(pts, LEFT_IRIS) or (left_e_cx, left_e_cy)
                    r_iris = iris_center_try(pts, RIGHT_IRIS) or (right_e_cx, right_e_cy)

                    # Eye sizes
                    l_left, l_top, l_right, l_bottom = eye_bbox(pts, LEFT_EYE_IDX)
                    r_left, r_top, r_right, r_bottom = eye_bbox(pts, RIGHT_EYE_IDX)

                    lw = max(1.0, l_right - l_left)
                    lh = max(1.0, l_bottom - l_top)
                    rw = max(1.0, r_right - r_left)
                    rh = max(1.0, r_bottom - r_top)

                    # Normalized offsets
                    off_lx = (l_iris[0] - left_e_cx) / lw
                    off_ly = (l_iris[1] - left_e_cy) / lh
                    off_rx = (r_iris[0] - right_e_cx) / rw
                    off_ry = (r_iris[1] - right_e_cy) / rh

                    # Average
                    off_x = (off_lx + off_rx) / 2.0
                    off_y = (off_ly + off_ry) / 2.0

                    # Smooth
                    sx, sy = self.smooth(off_x, off_y)

                    print(f"OFFSETS -> X: {sx:.3f}, Y: {sy:.3f}")

                    # Classify
                    gaze = self.classify_offset(sx, sy)

                    cv2.putText(frame, f"Gaze:{gaze}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                else:
                    gaze = "AWAY"

                # Show
                cv2.imshow("Calibrated Gaze Tracker", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                frame_no += 1

        cap.release()
        cv2.destroyAllWindows()


# -----------------------------
# Run directly
# -----------------------------
if __name__ == "__main__":
    gt = IrisOffsetGaze()
    gt.run(cam_index=0)
