import cv2

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise Exception("Camera not accessible. Try closing other apps or change index to 1.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.putText(frame, "Press ESC to quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
