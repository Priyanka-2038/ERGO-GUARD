import cv2
import mediapipe as mp
import pandas as pd
import joblib
import math

# ==============================
# ERGOGUARD - VIDEO PREDICTION
# ==============================

video_path = r"DATASET\video_env_1_anonymized\video_env_1_anonymized\ap_0016.mp4"
model_path = "models/ergoguard_model.pkl"

print("\n========================================")
print("     ERGOGUARD - VIDEO PREDICTION")
print("========================================")

# Load trained model
model = joblib.load(model_path)

print("✅ Model loaded successfully!")

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# --------------------------------
# Function to calculate angle
# --------------------------------
def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# --------------------------------
# Open video
# --------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video!")
    exit()

print("✅ Video opened successfully!")
print("Press Q to stop.\n")


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

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # -----------------------------
            # Get required body points
            # -----------------------------

            left_shoulder = [
                landmarks[11].x,
                landmarks[11].y
            ]

            right_shoulder = [
                landmarks[12].x,
                landmarks[12].y
            ]

            left_hip = [
                landmarks[23].x,
                landmarks[23].y
            ]

            right_hip = [
                landmarks[24].x,
                landmarks[24].y
            ]

            left_knee = [
                landmarks[25].x,
                landmarks[25].y
            ]

            right_knee = [
                landmarks[26].x,
                landmarks[26].y
            ]

            left_ankle = [
                landmarks[27].x,
                landmarks[27].y
            ]

            right_ankle = [
                landmarks[28].x,
                landmarks[28].y
            ]

            # -----------------------------
            # Calculate knee angles
            # -----------------------------

            left_knee_angle = calculate_angle(
                left_hip,
                left_knee,
                left_ankle
            )

            right_knee_angle = calculate_angle(
                right_hip,
                right_knee,
                right_ankle
            )

            average_knee_angle = (
                left_knee_angle +
                right_knee_angle
            ) / 2

            # -----------------------------
            # Calculate hip angles
            # -----------------------------

            left_hip_angle = calculate_angle(
                left_shoulder,
                left_hip,
                left_knee
            )

            right_hip_angle = calculate_angle(
                right_shoulder,
                right_hip,
                right_knee
            )

            # -----------------------------
            # Calculate trunk angle
            # -----------------------------

            shoulder_mid = [
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            ]

            hip_mid = [
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            ]

            vertical_point = [
                shoulder_mid[0],
                shoulder_mid[1] - 1
            ]

            trunk_angle = calculate_angle(
                vertical_point,
                shoulder_mid,
                hip_mid
            )

            # -----------------------------
            # Create ML input
            # -----------------------------

            features = pd.DataFrame([[
                left_knee_angle,
                right_knee_angle,
                average_knee_angle,
                left_hip_angle,
                right_hip_angle,
                trunk_angle
            ]], columns=[
                "left_knee_angle",
                "right_knee_angle",
                "average_knee_angle",
                "left_hip_angle",
                "right_hip_angle",
                "trunk_angle"
            ])

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction = model.predict(features)[0]

            probability = model.predict_proba(features)[0]

            confidence = max(probability) * 100

            if prediction == 0:

                label = "SAFE"
                status = "SAFE POSTURE"

            else:

                label = "UNSAFE"
                status = "WARNING: UNSAFE POSTURE!"

            # -----------------------------
            # Draw skeleton
            # -----------------------------

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # -----------------------------
            # Display prediction
            # -----------------------------

            cv2.putText(
                frame,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if label == "SAFE"
                else (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Trunk: {trunk_angle:.1f} deg",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Knee Avg: {average_knee_angle:.1f} deg",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Show video
        cv2.imshow(
            "ErgoGuard - Posture Prediction",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print("Prediction finished.")
print("========================================")