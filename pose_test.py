import cv2
import mediapipe as mp

video_path = r"DATASET\video_env_1_anonymized\video_env_1_anonymized\ap_0016.mp4"

cap = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect body landmarks
        results = pose.process(rgb_frame)

        # Draw the detected body landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("ErgoGuard - Pose Detection", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()