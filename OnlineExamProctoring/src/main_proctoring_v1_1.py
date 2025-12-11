import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque, defaultdict
from datetime import datetime

from event_queue import EventSender  # NEW


# ==========================================================
# Utilities
# ==========================================================
def now_ts():
    return time.time()


def now_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ==========================================================
# FINAL EVENT LOGGER (File + Queue for Backend)
# ==========================================================
class EventLogger:
    def __init__(self, sender=None):
        ensure_dir("logs")
        self.filename = f"logs/session_events_{now_str()}.jsonl"
        self.sender = sender
        print(f"[LOGGER] Writing events to {self.filename}")

    def log(self, event_type, frame_no, **kwargs):
        event = {
            "type": event_type,
            "frame": frame_no,
            "ts": now_ts()
        }
        event.update(kwargs)

        # Write to local session .jsonl
        with open(self.filename, "a") as f:
            f.write(json.dumps(event) + "\n")

        print("[EVENT]", event)

        # Send to backend (queue)
        if self.sender is not None:
            self.sender.enqueue(event)


def save_evidence_image(event_type, frame, frame_no):
    sub = os.path.join("logs", "evidence", event_type)
    ensure_dir(sub)
    fname = f"{sub}/{event_type}_{now_str()}_{frame_no}.jpg"
    cv2.imwrite(fname, frame)
    return fname


# ==========================================================
# CONFIG
# ==========================================================
CAM_INDEX = 0

# EAR thresholds
BLINK_EAR_THRESH = 0.20
LONG_CLOSE_EAR_THRESH = 0.18
LONG_CLOSE_FRAMES = 15

# MAR thresholds
TALK_MAR_THRESH = 0.15
RAPID_MAR_DELTA = 0.03
MIN_MAR_NOISE = 0.04
TALK_SUSTAINED_FRAMES = 2

# No face
AWAY_SUSTAINED_FRAMES = 12

# Gaze smoothing (used in combination: kalman -> alpha blend -> EMA)
SMOOTH_ALPHA = 0.45

# EMA alphas (NEW)
EAR_EMA_ALPHA = 0.30
MAR_EMA_ALPHA = 0.30
GAZE_EMA_ALPHA = 0.25

# Deadzone margin (for CENTER)
GAZE_MARGIN = 0.03

# Sustained gaze requirements
GAZE_NON_CENTER_SUSTAIN = 7   # left/right
GAZE_UP_SUSTAIN = 8           # up

# Backend
BACKEND_BASE_URL = "http://127.0.0.1:5000"
BACKEND_EVENTS_ENDPOINT = "/api/events"

# Mediapipe indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

MOUTH_IDX = [13, 14, 78, 308]


# ==========================================================
# Tiny Kalman
# ==========================================================
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


# ==========================================================
# Geometry helpers
# ==========================================================
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


def iris_center_try(pts, idx):
    try:
        arr = np.array([pts[i] for i in idx], dtype=np.float32)
        c = arr.mean(axis=0)
        return float(c[0]), float(c[1])
    except:
        return None


def normalize_point_in_box(p, box):
    left, top, right, bottom = box
    w = max(1, right - left)
    h = max(1, bottom - top)
    nx = (p[0] - left) / w
    ny = (p[1] - top) / h
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))


# ==========================================================
# Low-light enhancement
# ==========================================================
def enhance_low_light(frame, target_mean=110):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    if mean_val < target_mean:
        alpha = target_mean / (mean_val + 1e-3)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        return frame, True
    return frame, False


# ==========================================================
# PROCTOR ENGINE
# ==========================================================
class ProctoringEngineV11:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index

        # smoothing
        self.kx = TinyKalman()
        self.ky = TinyKalman()
        self.prev_x = None
        self.prev_y = None

        # EMA states (NEW)
        self.ear_ema = None
        self.mar_ema = None
        self.gaze_ema_x = None
        self.gaze_ema_y = None

        # counters
        self.closed_frame_count = 0
        self.talk_count = 0
        self.away_counter = 0
        self.frame_no = 0
        self.prev_mar = None

        # gaze counters
        self.gaze_lr_count = 0
        self.gaze_up_count = 0

        # sender + logger
        self.sender = EventSender(
            base_url=BACKEND_BASE_URL,
            endpoint=BACKEND_EVENTS_ENDPOINT,
            batch_size=10,
            send_interval=1.0,
            max_queue_size=2000,
        )
        self.sender.start()

        self.logger = EventLogger(sender=self.sender)

        # Mediapipe
        self.mpfm = mp.solutions.face_mesh
        self.fm_args = dict(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.load_gaze_config("config/gaze_config.json")


    # ----------------------------------------
    def load_gaze_config(self, path):
        with open(path, "r") as f:
            cfg = json.load(f)

        base_left = cfg["left_thresh"]
        base_right = cfg["right_thresh"]
        base_up = cfg["up_thresh"]

        self.left_thresh = min(1.0, base_left + GAZE_MARGIN)
        self.right_thresh = max(0.0, base_right - GAZE_MARGIN)
        self.up_thresh = max(0.0, base_up - GAZE_MARGIN)

        print("\n=== GAZE CALIBRATION (with margin) ===")
        print("base_left  =", base_left, " -> final:", self.left_thresh)
        print("base_right =", base_right, " -> final:", self.right_thresh)
        print("base_up    =", base_up, " -> final:", self.up_thresh)
        print("GAZE_MARGIN =", GAZE_MARGIN)
        print("======================================\n")

    # ----------------------------------------
    def smooth(self, nx, ny):
        """
        Existing tiny-Kalman + alpha blend smoothing.
        We'll then also apply an EMA to the final smoothed gaze to remove jitter.
        """
        sx = self.kx.update(nx)
        sy = self.ky.update(ny)

        if self.prev_x is None:
            ox, oy = sx, sy
        else:
            ox = SMOOTH_ALPHA * sx + (1 - SMOOTH_ALPHA) * self.prev_x
            oy = SMOOTH_ALPHA * sy + (1 - SMOOTH_ALPHA) * self.prev_y

        self.prev_x, self.prev_y = ox, oy

        # Gaze EMA (NEW)
        if self.gaze_ema_x is None:
            self.gaze_ema_x, self.gaze_ema_y = ox, oy
        else:
            self.gaze_ema_x = GAZE_EMA_ALPHA * ox + (1 - GAZE_EMA_ALPHA) * self.gaze_ema_x
            self.gaze_ema_y = GAZE_EMA_ALPHA * oy + (1 - GAZE_EMA_ALPHA) * self.gaze_ema_y

        # return the *ema-smoothed* gaze values (these are used for classification)
        return self.gaze_ema_x, self.gaze_ema_y

    # ----------------------------------------
    def classify_gaze(self, nx, ny):
        """
        Uses EMA-smoothed nx, ny values for stable classification.
        Horizontal: LEFT / RIGHT / CENTER
        Vertical: UP (if detected)
        """
        direction = "CENTER"
        if nx > self.left_thresh:
            direction = "LEFT"
        elif nx < self.right_thresh:
            direction = "RIGHT"

        # vertical check has higher priority (UP)
        if ny < self.up_thresh:
            return "UP"

        return direction

    # ----------------------------------------
    def annotate(self, frame, texts):
        y = 22
        for txt, color in texts:
            cv2.putText(frame, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

    # ----------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)

        with self.mpfm.FaceMesh(**self.fm_args) as fm:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame, boosted = enhance_low_light(frame)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                gaze_label = "AWAY"
                events = []

                # ------------------------------------------------------
                # FACE FOUND
                # ------------------------------------------------------
                if res.multi_face_landmarks:
                    face = res.multi_face_landmarks[0]
                    pts = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

                    # ---------------- EAR ----------------
                    ear_l = compute_EAR(pts, LEFT_EYE_IDX)
                    ear_r = compute_EAR(pts, RIGHT_EYE_IDX)
                    ear = (ear_l + ear_r) / 2.0

                    # EAR EMA (NEW) - smooth the ear readings to detect long closure more robustly
                    if self.ear_ema is None:
                        self.ear_ema = ear
                    else:
                        self.ear_ema = EAR_EMA_ALPHA * ear + (1 - EAR_EMA_ALPHA) * self.ear_ema

                    # Decide long closure using EMA ear (more stable)
                    if self.ear_ema < LONG_CLOSE_EAR_THRESH:
                        self.closed_frame_count += 1
                    else:
                        self.closed_frame_count = 0

                    if self.closed_frame_count >= LONG_CLOSE_FRAMES:
                        self.logger.log("LONG_EYE_CLOSURE", self.frame_no, ear=float(self.ear_ema))
                        save_evidence_image("long_closure", frame, self.frame_no)
                        events.append("LONG_EYE_CLOSURE")
                        self.closed_frame_count = 0

                    # ---------------- MAR ----------------
                    mar = compute_MAR(pts, MOUTH_IDX)

                    # MAR EMA (NEW) - stabilize mouth metric
                    if self.mar_ema is None:
                        self.mar_ema = mar
                    else:
                        self.mar_ema = MAR_EMA_ALPHA * mar + (1 - MAR_EMA_ALPHA) * self.mar_ema

                    # compute delta using raw mar vs previous raw (keeps some responsiveness)
                    if mar < MIN_MAR_NOISE:
                        mar_delta = 0.0
                    else:
                        mar_delta = 0.0 if self.prev_mar is None else abs(mar - self.prev_mar)

                    self.prev_mar = mar

                    talking_frame = False
                    # prefer EMA-mar for sustained mouth-open, plus fast delta for quick speaking
                    if self.mar_ema > TALK_MAR_THRESH:
                        talking_frame = True
                    elif mar > 0.12 and mar_delta > RAPID_MAR_DELTA:
                        talking_frame = True

                    if talking_frame:
                        self.talk_count += 1
                    else:
                        self.talk_count = 0

                    if self.talk_count >= TALK_SUSTAINED_FRAMES:
                        # log EMA-mar to backend for stability
                        self.logger.log("TALKING", self.frame_no, mar=float(self.mar_ema))
                        save_evidence_image("talking", frame, self.frame_no)
                        events.append("TALKING")
                        self.talk_count = 0

                    # ---------------- GAZE ----------------
                    lbox = eye_bbox(pts, LEFT_EYE_IDX)
                    rbox = eye_bbox(pts, RIGHT_EYE_IDX)

                    liris = iris_center_try(pts, LEFT_IRIS)
                    riris = iris_center_try(pts, RIGHT_IRIS)

                    if liris is None:
                        liris = ((lbox[0] + lbox[2]) / 2.0, (lbox[1] + lbox[3]) / 2.0)
                    if riris is None:
                        riris = ((rbox[0] + rbox[2]) / 2.0, (rbox[1] + rbox[3]) / 2.0)

                    lnx, lny = normalize_point_in_box(liris, lbox)
                    rnx, rny = normalize_point_in_box(riris, rbox)

                    nx = (lnx + rnx) / 2.0
                    ny = (lny + rny) / 2.0

                    # Pass through kalman + alpha + gaze EMA (smooth returns EMA values)
                    nx_ema, ny_ema = self.smooth(nx, ny)
                    gaze_label = self.classify_gaze(nx_ema, ny_ema)

                    # Track sustained non-center gaze
                    if gaze_label in ("LEFT", "RIGHT"):
                        self.gaze_lr_count += 1
                        self.gaze_up_count = 0
                    elif gaze_label == "UP":
                        self.gaze_up_count += 1
                        self.gaze_lr_count = 0
                    else:
                        self.gaze_lr_count = 0
                        self.gaze_up_count = 0

                    # Log sustained gaze events
                    if gaze_label in ("LEFT", "RIGHT") and \
                       self.gaze_lr_count == GAZE_NON_CENTER_SUSTAIN:
                        self.logger.log(f"GAZE_{gaze_label}", self.frame_no,
                                        nx=float(nx_ema), ny=float(ny_ema))
                        save_evidence_image(f"gaze_{gaze_label.lower()}", frame, self.frame_no)
                        events.append(f"GAZE_{gaze_label}")

                    # GAZE_UP uses EMA smoothed ny_ema (stable)
                    if gaze_label == "UP" and self.gaze_up_count == GAZE_UP_SUSTAIN:
                        self.logger.log("GAZE_UP", self.frame_no,
                                        nx=float(nx_ema), ny=float(ny_ema))
                        save_evidence_image("gaze_up", frame, self.frame_no)
                        events.append("GAZE_UP")

                    # ---------------- UI ----------------
                    texts = [
                        (f"EAR(ema): {self.ear_ema:.3f}", (0, 255, 0) if self.ear_ema >= LONG_CLOSE_EAR_THRESH else (0, 165, 255)),
                        (f"MAR(ema): {self.mar_ema:.3f}", (255, 255, 0)),
                        (f"Gaze: {gaze_label}", (0, 255, 255)),
                        (f"nx_ema: {nx_ema:.3f}", (200, 200, 200)),
                        (f"ny_ema: {ny_ema:.3f}", (200, 200, 200)),
                        (f"LR_cnt: {self.gaze_lr_count}", (180, 180, 180)),
                        (f"UP_cnt: {self.gaze_up_count}", (180, 180, 180)),
                    ]

                    if boosted:
                        texts.append(("LL: ON", (200, 200, 255)))

                    suspect = (
                        "LONG_EYE_CLOSURE" in events or
                        "TALKING" in events or
                        self.gaze_lr_count >= GAZE_NON_CENTER_SUSTAIN or
                        self.gaze_up_count >= GAZE_UP_SUSTAIN
                    )

                    texts.append((
                        "STATUS: " + ("SUSPECT" if suspect else "OK"),
                        (0, 0, 255) if suspect else (0, 200, 0)
                    ))

                    self.annotate(frame, texts)
                    self.away_counter = 0

                # ------------------------------------------------------
                # NO FACE
                # ------------------------------------------------------
                else:
                    self.away_counter += 1
                    texts = [("NO FACE DETECTED", (0, 0, 255))]
                    if self.away_counter >= AWAY_SUSTAINED_FRAMES:
                        self.logger.log("AWAY", self.frame_no)
                        save_evidence_image("away", frame, self.frame_no)
                        texts.append(("STATUS: SUSPECT", (0, 0, 255)))
                        self.away_counter = 0
                    else:
                        texts.append(("STATUS: OK", (0, 200, 0)))
                    self.annotate(frame, texts)

                cv2.imshow("Proctoring v1.1", frame)
                self.frame_no += 1

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.sender.stop()


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":
    engine = ProctoringEngineV11(cam_index=CAM_INDEX)
    engine.run()
