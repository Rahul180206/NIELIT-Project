import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
MOUTH = [13, 14, 78, 308]

def eu(a, b):
    return np.linalg.norm(a - b)

def mar(pts):
    up = np.array(pts[MOUTH[0]])
    low = np.array(pts[MOUTH[1]])
    left = np.array(pts[MOUTH[2]])
    right = np.array(pts[MOUTH[3]])
    return eu(up, low) / eu(left, right)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(refine_landmarks=True) as fm:
    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            pts = [(int(l.x*w), int(l.y*h)) for l in face.landmark]

            v = mar(pts)

            print(f"MAR = {v:.3f}")

            cv2.putText(frame, f"MAR={v:.3f}", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("MAR Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
