import cv2
import numpy as np


def optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Не вдалося зчитати перший кадр.")
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    print("[INFO] Запуск Optical Flow. Натисніть 'q' для виходу.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Optical Flow", bgr_flow)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()


def background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    print("[INFO] Запуск Background Subtraction. Натисніть 'q' для виходу.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        cv2.imshow("Background Subtraction", fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = "bird1.mp4"

optical_flow(video_path)
background_subtraction(video_path)
