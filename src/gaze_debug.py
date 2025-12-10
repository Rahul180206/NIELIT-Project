import cv2
import mediapipe as mp
import numpy as np

mpfm = mp.solutions.face_mesh

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
LEFT_IRIS = [468,469,470,471]
RIGHT_IRIS = [473,474,475,476]

def iris_center(pts, idx):
    arr = np.array([pts[i] for i in idx], dtype=np.float32)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1])

def eye_box(pts, idx):
    xs = [pts[i][0] for i in idx]
    ys = [pts[i][1] for i in idx]
    return min(xs), min(ys), max(xs), max(ys)

def normalize(p, box):
    left, top, right, bottom = box
    w = max(1, right-left)
    h = max(1, bottom-top)
    nx = (p[0] - left) / w
    ny = (p[1] - top) / h
    return nx, ny

cap = cv2.VideoCapture(0)

with mpfm.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            pts = [(int(l.x * w), int(l.y * h)) for l in res.multi_face_landmarks[0].landmark]

            li = iris_center(pts, LEFT_IRIS)
            ri = iris_center(pts, RIGHT_IRIS)

            lb = eye_box(pts, LEFT_EYE)
            rb = eye_box(pts, RIGHT_EYE)

            lnx, lny = normalize(li, lb)
            rnx, rny = normalize(ri, rb)

            nx = (lnx + rnx) / 2
            ny = (lny + rny) / 2

            cv2.putText(frame, f"NX={nx:.3f}  NY={ny:.3f}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Gaze Debug", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
