# src/proctor.py
# Integrated Proctoring Orchestrator
# Combines EAR blink, MAR talking, and calibrated iris-offset gaze tracker
# Single webcam/FaceMesh loop. Saves evidence and logs JSONL events.

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque

# -----------------------
# Config / thresholds (tuned for your webcam)
# -----------------------
BLINK_EAR_THRESH = 0.20        # short blink
LONG_CLOSE_EAR_THRESH = 0.18   # long eye closure
LONG_CLOSE_FRAMES = 12         # sustained frames for long closure

TALK_MAR_THRESH = 0.33         # MAR threshold (talk)
TALK_FRAMES = 2                # sustained frames for talking

GAZE_SUSTAINED_FRAMES = 10     # frames before logging gaze event
GAZE_X_THRESH = 0.06           # left/right threshold (calibrated)
GAZE_UP_Y_THRESH = -0.28       # up threshold (calibrated)

EVIDENCE_DIR = "logs/evidence"
LOG_PATH = "logs/proctor_events.jsonl"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# -----------------------
# Mediapipe indices
# -----------------------
LEFT_EYE_IDX  = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
LEFT_IRIS     = [468,469,470,471]
RIGHT_IRIS    = [473,474,475,476]
MOUTH_IDX     = [13, 14, 78, 308]   # up, low, left, right for MAR (approx)

# -----------------------
# Tiny 1D Kalman for smoothing offsets
# -----------------------
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

# -----------------------
# Helpers (EAR, MAR, geometry)
# -----------------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

def compute_EAR(pts, eye_idx):
    p = np.array([pts[i] for i in eye_idx], dtype=np.float32)
    A = euclidean(p[1], p[5])
    B = euclidean(p[2], p[4])
    C = euclidean(p[0], p[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def compute_MAR(pts, mouth_idx):
    up = np.array(pts[mouth_idx[0]])
    low = np.array(pts[mouth_idx[1]])
    left = np.array(pts[mouth_idx[2]])
    right = np.array(pts[mouth_idx[3]])
    vertical = euclidean(up, low)
    horizontal = euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0

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
    print("[EVENT]", ev)

# -----------------------
# Orchestrator main class
# -----------------------
class Proctor:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index
        self.kx = TinyKalman()
        self.ky = TinyKalman()
        self.alpha = 0.45
        self.prev_x = None
        self.prev_y = None

        # counters for sustained events
        self.closed_frame_count = 0
        self.talk_count = 0

        self.gaze_counters = {"LEFT":0, "RIGHT":0, "UP":0, "CENTER":0, "AWAY":0}
        self.frame_no = 0

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

    def classify_gaze(self, sx, sy):
        # horizontal
        if sx < -GAZE_X_THRESH:
            return "LEFT"
        if sx > GAZE_X_THRESH:
            return "RIGHT"
        # vertical: only UP is reliable for your webcam
        if sy < GAZE_UP_Y_THRESH:
            return "UP"
        return "CENTER"

    def run(self):
        mpfm = mp.solutions.face_mesh
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        with mpfm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                # defaults
                gaze_label = "AWAY"
                ear_text = ""
                mar_text = ""
                combined_alert = False

                if res.multi_face_landmarks:
                    face = res.multi_face_landmarks[0]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

                    # ---------------- EAR (blink / long closure)
                    ear_l = compute_EAR(pts, LEFT_EYE_IDX)
                    ear_r = compute_EAR(pts, RIGHT_EYE_IDX)
                    ear = (ear_l + ear_r) / 2.0
                    ear_text = f"EAR:{ear:.3f}"

                    if ear < LONG_CLOSE_EAR_THRESH:
                        self.closed_frame_count += 1
                    else:
                        self.closed_frame_count = 0

                    if self.closed_frame_count >= LONG_CLOSE_FRAMES:
                        # long eye closure event
                        ev = {"state":"LONG_EYE_CLOSURE", "frame": self.frame_no, "ear": float(ear), "ts": time.time()}
                        save_event(ev)
                        fname = f"{EVIDENCE_DIR}/long_closure_{self.frame_no}.jpg"
                        cv2.imwrite(fname, frame)
                        ear_text += " [LONG_CLOSE]"
                        combined_alert = True

                    # draw EAR overlay
                    cv2.putText(frame, ear_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # ---------------- MAR (talking)
                    mar = compute_MAR(pts, MOUTH_IDX)
                    mar_text = f"MAR:{mar:.3f}"

                    # rapid change detection not required here since small TALK_FRAMES used
                    if mar > TALK_MAR_THRESH:
                        self.talk_count += 1
                    else:
                        self.talk_count = 0

                    if self.talk_count >= TALK_FRAMES:
                        ev = {"state":"TALKING", "frame": self.frame_no, "mar": float(mar), "ts": time.time()}
                        save_event(ev)
                        fname = f"{EVIDENCE_DIR}/talking_{self.frame_no}.jpg"
                        cv2.imwrite(fname, frame)
                        mar_text += " [TALK]"
                        combined_alert = True

                    cv2.putText(frame, mar_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                    # ---------------- GAZE (iris offset)
                    # compute eye centers & iris centers
                    left_cx, left_cy = center_of_points(pts, LEFT_EYE_IDX)
                    right_cx, right_cy = center_of_points(pts, RIGHT_EYE_IDX)

                    l_iris = iris_center_try(pts, LEFT_IRIS) or (left_cx, left_cy)
                    r_iris = iris_center_try(pts, RIGHT_IRIS) or (right_cx, right_cy)

                    # eye boxes sizes
                    l_l, l_t, l_r, l_b = eye_bbox(pts, LEFT_EYE_IDX)
                    r_l, r_t, r_r, r_b = eye_bbox(pts, RIGHT_EYE_IDX)
                    lw = max(1.0, l_r - l_l)
                    lh = max(1.0, l_b - l_t)
                    rw = max(1.0, r_r - r_l)
                    rh = max(1.0, r_b - r_t)

                    off_lx = (l_iris[0] - left_cx) / lw
                    off_ly = (l_iris[1] - left_cy) / lh
                    off_rx = (r_iris[0] - right_cx) / rw
                    off_ry = (r_iris[1] - right_cy) / rh

                    off_x = (off_lx + off_rx) / 2.0
                    off_y = (off_ly + off_ry) / 2.0

                    sx, sy = self.smooth(off_x, off_y)

                    # debug print (optional)
                    # print(f"OFFSETS -> X: {sx:.3f}, Y: {sy:.3f}")

                    gaze_label = self.classify_gaze(sx, sy)

                    # sustained counters for gaze
                    for k in self.gaze_counters.keys():
                        if k == gaze_label:
                            self.gaze_counters[k] += 1
                        else:
                            self.gaze_counters[k] = 0

                    # when sustained, save evidence & log (ignore CENTER/AWAY)
                    if gaze_label in self.gaze_counters and self.gaze_counters[gaze_label] == GAZE_SUSTAINED_FRAMES and gaze_label not in ("CENTER", "AWAY"):
                        ev = {"state": f"GAZE_{gaze_label}", "frame": self.frame_no, "offx": float(sx), "offy": float(sy), "ts": time.time()}
                        save_event(ev)
                        fname = f"{EVIDENCE_DIR}/gaze_{gaze_label}_{self.frame_no}.jpg"
                        cv2.imwrite(fname, frame)
                        combined_alert = True

                    # draw gaze overlay and small markers
                    cv2.putText(frame, f"Gaze:{gaze_label}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    # draw eye centers and iris points
                    cv2.circle(frame, (int(left_cx), int(left_cy)), 2, (0,255,0), -1)
                    cv2.circle(frame, (int(right_cx), int(right_cy)), 2, (0,255,0), -1)
                    cv2.circle(frame, (int(l_iris[0]), int(l_iris[1])), 2, (0,200,0), -1)
                    cv2.circle(frame, (int(r_iris[0]), int(r_iris[1])), 2, (0,200,0), -1)

                else:
                    # NO face
                    gaze_label = "AWAY"
                    # increment away counter and log if sustained
                    self.gaze_counters = {k: (self.gaze_counters[k] + 1 if k == "AWAY" else 0) for k in self.gaze_counters}
                    if self.gaze_counters["AWAY"] == GAZE_SUSTAINED_FRAMES:
                        ev = {"state":"AWAY", "frame": self.frame_no, "ts": time.time()}
                        save_event(ev)
                        fname = f"{EVIDENCE_DIR}/away_{self.frame_no}.jpg"
                        cv2.imwrite(fname, frame)
                        combined_alert = True

                # Combined cheating indicator
                status_text = "OK"
                if combined_alert:
                    status_text = "CHEATING_SUSPECTED"

                cv2.putText(frame, f"STATUS: {status_text}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255) if combined_alert else (0,200,0), 2)

                cv2.imshow("Proctor - EAR/MAR/Gaze Integrated", frame)

                self.frame_no += 1
                # simple exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    p = Proctor(cam_index=0)
    p.run()
