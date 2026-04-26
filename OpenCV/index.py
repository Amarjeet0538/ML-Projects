import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Hand landmark connections (same as MediaPipe skeleton)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def run_hand_tracking_on_webcam():

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cam = cv2.VideoCapture(0)
    frame_timestamp = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        while cam.isOpened():
            success, frame = cam.read()

            if not success:
                print("Empty frame! Skipping.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            results = landmarker.detect_for_video(mp_image, frame_timestamp)
            frame_timestamp += 1

            h, w, _ = frame.shape

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:

                    # Convert normalized coordinates → pixel coordinates
                    points = []
                    for lm in hand_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    # Draw skeleton lines
                    for connection in HAND_CONNECTIONS:
                        start = points[connection[0]]
                        end = points[connection[1]]
                        cv2.line(frame, start, end, (0, 255, 0), 2)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_tracking_on_webcam()
