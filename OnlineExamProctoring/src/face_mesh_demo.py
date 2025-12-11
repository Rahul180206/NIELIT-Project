import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

FACE_MESH_ARGS = dict(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_face_mesh.FaceMesh(**FACE_MESH_ARGS) as face_mesh:
        sample_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION
                    )

            cv2.putText(
                frame,
                "Press 'S' to save sample image | ESC to exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Face Mesh Demo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == ord('s'):
                save_path = f"data/face_mesh_sample_{sample_id}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"[Saved] {save_path}")
                sample_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
