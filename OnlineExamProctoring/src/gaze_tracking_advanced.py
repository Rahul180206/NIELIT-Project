import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

from collections import deque

# -----------------------------
# Tiny 1D Kalman Filter
# -----------------------------
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

# Eye indices
LEFT_EYE_IDX  = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]

# Iris indices (mediapipe refine_landmarks=True)
LEFT_IRIS  = [468,469,470,471]
RIGHT_IRIS = [473,474,475,476]


class AdvancedGaze:

    def __init__(self,
                 sustained_frames=12,
                 smooth_alpha=0.4):
        self.mp_face = mp.solutions.face_mesh
        self.kx = TinyKalman()
        self.ky = TinyKalman()
        self.alpha = smooth_alpha
        self.prev_x = None
        self.prev_y = None

        self.sustained_frames = sustained_frames
        self.counters = {"LEFT":0, "RIGHT":0, "UP":0, "DOWN":0,
                         "CENTER":0, "AWAY":0}

        self.history = deque(maxlen=10)

        # load calibration
        self.config = self.load_config("config/gaze_config.json")

        # auto-correct thresholds
        self.correct_thresholds()

        os.makedirs("logs/evidence", exist_ok=True)

    # ----------------------------------------------
    # Load calibration JSON
    # ----------------------------------------------
    def load_config(self, path):
        with open(path, "r") as f:
            cfg = json.load(f)
        return cfg

    # ----------------------------------------------
    # Auto-detect reversed thresholds and fix
    # ----------------------------------------------
    def correct_thresholds(self):
        c = self.config

        lt = c["left_thresh"]
        rt = c["right_thresh"]
        ut = c["up_thresh"]
        dt = c["down_thresh"]

        # clamp negative values
        if dt < 0:
            dt = max(dt, 0.05)
        if ut < 0:
            ut = max(ut, 0.05)

        # AUTO FIX 1: Horizontal Swap
        # Normally: left_thresh < right_thresh
        # If reversed: swap logic
        self.horizontal_reversed = False
        if lt > rt:
            self.horizontal_reversed = True

        # AUTO FIX 2: Vertical Swap
        # Normally: up_thresh < down_thresh
        self.vertical_reversed = False
        if ut > dt:
            self.vertical_reversed = True

        self.left_thresh  = min(lt, rt)
        self.right_thresh = max(lt, rt)
        self.up_thresh    = min(ut, dt)
        self.down_thresh  = max(ut, dt)

        print("\n=== AUTO-CORRECTED THRESHOLDS ===")
        print(f"LEFT  threshold : {self.left_thresh}")
        print(f"RIGHT threshold : {self.right_thresh}")
        print(f"UP    threshold : {self.up_thresh}")
        print(f"DOWN  threshold : {self.down_thresh}")

        print(f"\nHorizontal reversed? {self.horizontal_reversed}")
        print(f"Vertical reversed?   {self.vertical_reversed}")
        print("=================================\n")


    # ----------------------------------------------
    # Helper functions
    # ----------------------------------------------
    def eye_box(self, pts, idx):
        xs = [pts[i][0] for i in idx]
        ys = [pts[i][1] for i in idx]
        return (min(xs), min(ys), max(xs), max(ys))

    def iris_center(self, pts, idx):
        pts = np.array([pts[i] for i in idx])
        c = pts.mean(axis=0)
        return int(c[0]), int(c[1])

    def normalize(self, p, box):
        left, top, right, bottom = box
        w = max(1, right-left)
        h = max(1, bottom-top)
        nx = (p[0] - left) / w
        ny = (p[1] - top) / h
        nx = max(0,min(1,nx))
        ny = max(0,min(1,ny))
        return nx, ny

    def smooth(self, x, y):
        kx = self.kx.update(x)
        ky = self.ky.update(y)

        if self.prev_x is None:
            sx, sy = kx, ky
        else:
            sx = self.alpha*kx + (1-self.alpha)*self.prev_x
            sy = self.alpha*ky + (1-self.alpha)*self.prev_y

        self.prev_x, self.prev_y = sx, sy
        return sx, sy

    # ----------------------------------------------
    # Main classification logic with auto-correction
    # ----------------------------------------------
    def classify(self, nx, ny):
        label_h = "CENTER"
        label_v = "CENTER"

        # horizontal
        if nx < self.left_thresh:
            label_h = "LEFT"
        elif nx > self.right_thresh:
            label_h = "RIGHT"

        # vertical
        if ny < self.up_thresh:
            label_v = "UP"
        elif ny > self.down_thresh:
            label_v = "DOWN"

        # apply auto swap
        if self.horizontal_reversed:
            if label_h == "LEFT": label_h = "RIGHT"
            elif label_h == "RIGHT": label_h = "LEFT"

        if self.vertical_reversed:
            if label_v == "UP": label_v = "DOWN"
            elif label_v == "DOWN": label_v = "UP"

        # final combined result
        if label_h == "CENTER" and label_v == "CENTER":
            return "CENTER"
        if label_h != "CENTER" and label_v == "CENTER":
            return label_h
        if label_h == "CENTER" and label_v != "CENTER":
            return label_v

        return f"{label_v}-{label_h}"

    # ----------------------------------------------
    # Main run
    # ----------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        with self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as fm:

            frame_no = 0

            while True:
                r, frame = cap.read()
                if not r: break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                h, w = frame.shape[:2]

                if res.multi_face_landmarks:
                    face = res.multi_face_landmarks[0]
                    pts = [(int(lm.x*w), int(lm.y*h)) for lm in face.landmark]

                    # left iris
                    li = self.iris_center(pts, LEFT_IRIS)
                    ri = self.iris_center(pts, RIGHT_IRIS)

                    # eye boxes
                    lb = self.eye_box(pts, LEFT_EYE_IDX)
                    rb = self.eye_box(pts, RIGHT_EYE_IDX)

                    lnx, lny = self.normalize(li, lb)
                    rnx, rny = self.normalize(ri, rb)

                    nx = (lnx + rnx)/2
                    ny = (lny + rny)/2

                    nx, ny = self.smooth(nx, ny)
                    label = self.classify(nx, ny)

                    cv2.putText(frame, f"Gaze: {label}", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    cv2.putText(frame, f"nx={nx:.2f} ny={ny:.2f}", (20,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

                else:
                    label = "AWAY"
                    cv2.putText(frame, "NO FACE", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.imshow("Corrected Gaze Tracker", frame)
                frame_no += 1

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    ag = AdvancedGaze()
    ag.run()
