# src/main_proctoring.py
# Proctoring v1.0 - Integrated EAR / MAR / Calibrated Gaze / Face-Away / Attention Score
# Single FaceMesh loop. JSONL logging + annotated evidence images organized by event type.

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque, defaultdict
from datetime import datetime

# -------------------------
# CONFIG (tweak these as needed)
# -------------------------
CAM_INDEX = 0

# EAR thresholds
BLINK_EAR_THRESH = 0.20
LONG_CLOSE_EAR_THRESH = 0.18
LONG_CLOSE_FRAMES = 12   # ~0.4s @30fps

# MAR thresholds
TALK_MAR_THRESH = 0.25
TALK_FRAMES = 2

# Gaze thresholds (calibrated for your camera)
GAZE_SUSTAINED_FRAMES = 10
GAZE_X_THRESH = 0.035
GAZE_UP_Y_THRESH = -0.24

# Face-away sustained frames
AWAY_SUSTAINED_FRAMES = 12

# Smoothing
SMOOTH_ALPHA = 0.45

# Logging / evidence
LOG_PATH = "logs/proctor_events.jsonl"
EVIDENCE_DIR = "logs/evidence"
SUMMARY_DIR = "logs"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Mediapipe indices
LEFT_EYE_IDX  = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
LEFT_IRIS     = [468,469,470,471]
RIGHT_IRIS    = [473,474,475,476]
MOUTH_IDX     = [13, 14, 78, 308]  # up, low, left, right

# -------------------------
# Utilities
# -------------------------
def now_ts():
    return time.time()

def now_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_event(ev):
    ensure_dir(os.path.dirname(LOG_PATH))
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(ev) + "\n")
    print("[EVENT]", ev)

def save_evidence_image(event_type, frame, frame_no):
    sub = os.path.join(EVIDENCE_DIR, event_type)
    ensure_dir(sub)
    fname = f"{sub}/{event_type}_{now_str()}_{frame_no}.jpg"
    cv2.imwrite(fname, frame)
    return fname

# -------------------------
# Tiny 1D Kalman smoother (for iris offsets)
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

# -------------------------
# Geometric helpers
# -------------------------
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

# -------------------------
# Proctor engine
# -------------------------
class ProctoringEngine:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index
        self.kx = TinyKalman()
        self.ky = TinyKalman()
        self.alpha = SMOOTH_ALPHA
        self.prev_x = None
        self.prev_y = None

        # counters
        self.closed_frame_count = 0
        self.talk_count = 0
        self.gaze_counters = {"LEFT":0, "RIGHT":0, "UP":0, "CENTER":0, "AWAY":0}
        self.away_counter = 0

        # session stats
        self.stats = defaultdict(int)
        self.frame_no = 0
        self.history = deque(maxlen=30)

        # FaceMesh
        self.mpfm = mp.solutions.face_mesh
        self.fm_args = dict(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        if sx < -GAZE_X_THRESH:
            return "LEFT"
        if sx > GAZE_X_THRESH:
            return "RIGHT"
        if sy < GAZE_UP_Y_THRESH:
            return "UP"
        return "CENTER"

    def attention_score(self, gaze_label, ear_ok, talking):
        # Simple weighted attention:
        # center gaze + eyes open + not talking -> 1.0 (best)
        score = 0.0
        # gaze weight
        if gaze_label == "CENTER":
            score += 0.6
        elif gaze_label == "UP":
            score += 0.3
        else: # left/right/away
            score += 0.15
        # eyes
        score += 0.25 if ear_ok else 0.0
        # talking reduces attention
        if talking:
            score -= 0.15
        # clamp
        return max(0.0, min(1.0, score))

    def annotate_frame(self, frame, texts, pts_draw=[]):
        y = 20
        for t, color in texts:
            cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
        for p in pts_draw:
            cv2.circle(frame, p, 2, (0,255,0), -1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        with self.mpfm.FaceMesh(**self.fm_args) as fm:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                # default labels & flags
                gaze_label = "AWAY"
                ear_ok = True
                talking = False
                event_logged = []

                if res.multi_face_landmarks:
                    face = res.multi_face_landmarks[0]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

                    # EAR
                    ear_l = compute_EAR(pts, LEFT_EYE_IDX)
                    ear_r = compute_EAR(pts, RIGHT_EYE_IDX)
                    ear = (ear_l + ear_r) / 2.0
                    ear_ok = ear >= LONG_CLOSE_EAR_THRESH

                    # long closure detection
                    if ear < LONG_CLOSE_EAR_THRESH:
                        self.closed_frame_count += 1
                    else:
                        if 0 < self.closed_frame_count < LONG_CLOSE_FRAMES:
                            self.stats["short_closures"] += 1
                        self.closed_frame_count = 0

                    if self.closed_frame_count >= LONG_CLOSE_FRAMES:
                        self.stats["long_closures"] += 1
                        ev = {"state":"LONG_EYE_CLOSURE", "frame": self.frame_no, "ear": float(ear), "ts": now_ts()}
                        save_event(ev)
                        save_evidence_image("long_closure", frame, self.frame_no)
                        event_logged.append("LONG_EYE_CLOSURE")
                        # reset counter so repeated saving is avoided for same sustained period
                        self.closed_frame_count = 0

                    # MAR / talking
                    mar = compute_MAR(pts, MOUTH_IDX)
                    if mar > TALK_MAR_THRESH:
                        self.talk_count += 1
                    else:
                        if 0 < self.talk_count < TALK_FRAMES:
                            self.stats["short_talks"] += 1
                        self.talk_count = 0

                    if self.talk_count >= TALK_FRAMES:
                        talking = True
                        self.stats["talking_events"] += 1
                        ev = {"state":"TALKING", "frame": self.frame_no, "mar": float(mar), "ts": now_ts()}
                        save_event(ev)
                        save_evidence_image("talking", frame, self.frame_no)
                        event_logged.append("TALKING")
                        self.talk_count = 0

                    # Gaze offsets
                    left_cx, left_cy = center_of_points(pts, LEFT_EYE_IDX)
                    right_cx, right_cy = center_of_points(pts, RIGHT_EYE_IDX)
                    l_iris = iris_center_try(pts, LEFT_IRIS) or (left_cx, left_cy)
                    r_iris = iris_center_try(pts, RIGHT_IRIS) or (right_cx, right_cy)
                    l_l, l_t, l_r, l_b = eye_bbox(pts, LEFT_EYE_IDX)
                    r_l, r_t, r_r, r_b = eye_bbox(pts, RIGHT_EYE_IDX)
                    lw = max(1.0, l_r - l_l); lh = max(1.0, l_b - l_t)
                    rw = max(1.0, r_r - r_l); rh = max(1.0, r_b - r_t)
                    off_lx = (l_iris[0] - left_cx) / lw
                    off_ly = (l_iris[1] - left_cy) / lh
                    off_rx = (r_iris[0] - right_cx) / rw
                    off_ry = (r_iris[1] - right_cy) / rh
                    off_x = (off_lx + off_rx) / 2.0
                    off_y = (off_ly + off_ry) / 2.0
                    sx, sy = self.smooth(off_x, off_y)
                    gaze_label = self.classify_gaze(sx, sy)

                    # gaze sustained logging
                    for k in self.gaze_counters.keys():
                        if k == gaze_label:
                            self.gaze_counters[k] += 1
                        else:
                            self.gaze_counters[k] = 0

                    if gaze_label not in ("CENTER", "AWAY") and self.gaze_counters[gaze_label] >= GAZE_SUSTAINED_FRAMES:
                        self.stats["gaze_events_"+gaze_label.lower()] += 1
                        ev = {"state":f"GAZE_{gaze_label}", "frame": self.frame_no, "offx": float(sx), "offy": float(sy), "ts": now_ts()}
                        save_event(ev)
                        save_evidence_image(f"gaze_{gaze_label.lower()}", frame, self.frame_no)
                        event_logged.append(f"GAZE_{gaze_label}")
                        self.gaze_counters[gaze_label] = 0

                    # face present, reset away counter
                    self.away_counter = 0

                    # annotate debug points
                    pts_draw = [(int(left_cx), int(left_cy)), (int(right_cx), int(right_cy)),
                                (int(l_iris[0]), int(l_iris[1])), (int(r_iris[0]), int(r_iris[1]))]

                    # overlays
                    texts = [
                        (f"EAR:{ear:.3f}", (0,255,0)),
                        (f"MAR:{mar:.3f}", (255,255,0)),
                        (f"Gaze:{gaze_label}", (0,255,255))
                    ]
                    # attention score
                    att = self.attention_score(gaze_label, ear_ok, talking)
                    texts.append((f"ATT:{att:.2f}", (200,200,200)))
                    status = "OK" if att >= 0.6 and not event_logged else "SUSPECT"
                    texts.append((f"STATUS:{status}", (0,200,0) if status=="OK" else (0,0,255)))

                    self.annotate_frame(frame, texts, pts_draw)

                else:
                    # no face detected
                    self.away_counter += 1
                    if self.away_counter >= AWAY_SUSTAINED_FRAMES:
                        self.stats["away_events"] += 1
                        ev = {"state":"AWAY", "frame": self.frame_no, "ts": now_ts()}
                        save_event(ev)
                        save_evidence_image("away", frame, self.frame_no)
                        self.away_counter = 0
                    texts = [(f"NO FACE DETECTED", (0,0,255))]
                    self.annotate_frame(frame, texts)

                # incremental stats
                self.history.append((self.frame_no, gaze_label, now_ts()))
                self.frame_no += 1

                cv2.imshow("Proctoring v1.0 - Press ESC to exit", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.save_summary()

    def save_summary(self):
        summary = {
            "timestamp": now_str(),
            "total_frames": int(self.frame_no),
            "stats": dict(self.stats)
        }
        fname = f"{SUMMARY_DIR}/run_summary_{now_str()}.json"
        with open(fname, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUMMARY SAVED] {fname}")
        return fname

# -------------------------
# Run script
# -------------------------
if __name__ == "__main__":
    engine = ProctoringEngine(cam_index=CAM_INDEX)
    engine.run()
