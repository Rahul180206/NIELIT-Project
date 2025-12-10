import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (Mediapipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def compute_EAR(landmarks, eye_indices):
    pts = np.array([landmarks[i] for i in eye_indices])

    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])

    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # thresholds
    BLINK_THRESH = 0.20
    LONG_CLOSE_THRESH = 0.18
    LONG_CLOSE_FRAMES = 12   # ~0.4 seconds at 30 FPS

    closed_frame_count = 0

    # folder for evidence
    os.makedirs("logs/evidence", exist_ok=True)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                points = []
                for lm in face.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))

                ear_left = compute_EAR(points, LEFT_EYE)
                ear_right = compute_EAR(points, RIGHT_EYE)
                ear_avg = (ear_left + ear_right) / 2.0

                # EAR overlay
                cv2.putText(frame, f"EAR: {ear_avg:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Blink (short closure)
                if ear_avg < BLINK_THRESH:
                    cv2.putText(frame, "Blink / Eyes Closed", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                # Long closure detection
                if ear_avg < LONG_CLOSE_THRESH:
                    closed_frame_count += 1
                else:
                    closed_frame_count = 0

                if closed_frame_count >= LONG_CLOSE_FRAMES:
                    cv2.putText(frame, "LONG EYE CLOSURE DETECTED!", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

                    print(f"[ALERT] Long closure at frame {frame_no}, EAR={ear_avg:.3f}")

                    # Save evidence image
                    filename = f"logs/evidence/long_closure_{frame_no}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[Saved] {filename}")

            cv2.imshow("EAR Blink + Long Closure", frame)
            frame_no += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
