import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

MOUTH = [13, 14, 78, 308]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def compute_MAR(landmarks, idx):
    up = np.array(landmarks[idx[0]])
    low = np.array(landmarks[idx[1]])
    left = np.array(landmarks[idx[2]])
    right = np.array(landmarks[idx[3]])

    vertical = euclidean(up, low)
    horizontal = euclidean(left, right)
    return vertical / horizontal if horizontal != 0 else 0

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    TALK_THRESH = 0.33            # NEW LOWER THRESHOLD
    LONG_TALK_FRAMES = 2           # FASTER DETECTION
    RAPID_OSC_DIFF = 0.05          # MAR change required to classify rapid motion

    talk_count = 0
    frame_no = 0
    previous_mar = 0

    os.makedirs("logs/evidence", exist_ok=True)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
                mar = compute_MAR(pts, MOUTH)

                cv2.putText(frame, f"MAR: {mar:.3f}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                # Rapid MAR change (talking motion)
                mar_change = abs(mar - previous_mar)

                # Talking classification (two conditions)
                if mar > TALK_THRESH or mar_change > RAPID_OSC_DIFF:
                    talk_count += 1
                else:
                    talk_count = 0

                if talk_count >= LONG_TALK_FRAMES:
                    cv2.putText(frame, "TALKING DETECTED", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                    print(f"[ALERT] Talking at frame {frame_no}, MAR={mar:.3f}")

                    path = f"logs/evidence/talking_{frame_no}.jpg"
                    cv2.imwrite(path, frame)

                previous_mar = mar

            cv2.imshow("MAR Talking Detection", frame)
            frame_no += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
