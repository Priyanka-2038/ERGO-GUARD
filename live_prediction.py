import cv2
import mediapipe as mp
import pandas as pd
import joblib
import math

print("\n========================================")
print("     ERGOGUARD - MULTI-CLASS WEBCAM")
print("========================================")


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/ergoguard_model.pkl")

print("✅ Multi-class model loaded successfully!")


# ==========================================
# LABEL MAPPING
# ==========================================

label_mapping = {
    0: "SAFE",
    1: "EXCESSIVE_FORWARD_BENDING",
    2: "BENDING_WITH_STRAIGHT_KNEES",
    3: "ROUNDED_BACK",
    4: "TWISTED_TRUNK",
    5: "SIDE_BENDING",
    6: "ASYMMETRICAL_LIFTING",
    7: "EXCESSIVE_SQUATTING",
    8: "OVERHEAD_OR_REACHING"
}


# ==========================================
# MEDIAPIPE
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# ANGLE FUNCTIONS
# ==========================================

def calculate_angle(a, b, c):

    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]

    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    denominator = (
        math.sqrt(ba_x ** 2 + ba_y ** 2) *
        math.sqrt(bc_x ** 2 + bc_y ** 2)
    )

    if denominator == 0:
        return 0.0

    cosine_angle = (
        ba_x * bc_x + ba_y * bc_y
    ) / denominator

    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def angle_from_vertical(shoulder_mid, hip_mid):

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    return math.degrees(
        math.atan2(abs(dx), abs(dy))
    )


def line_angle(point_a, point_b):

    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]

    return math.degrees(math.atan2(dy, dx))


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

print("✅ Webcam started!")
print("Press Q to quit.")


# ==========================================
# POSE DETECTION
# ==========================================

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("❌ Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # ==================================
            # BODY LANDMARKS
            # ==================================

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

            left_elbow = [
                landmarks[13].x,
                landmarks[13].y
            ]

            right_elbow = [
                landmarks[14].x,
                landmarks[14].y
            ]

            left_wrist = [
                landmarks[15].x,
                landmarks[15].y
            ]

            right_wrist = [
                landmarks[16].x,
                landmarks[16].y
            ]

            # ==================================
            # KNEE FEATURES
            # ==================================

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

            # ==================================
            # HIP FEATURES
            # ==================================

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

            # ==================================
            # MIDPOINTS
            # ==================================

            shoulder_mid = [
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            ]

            hip_mid = [
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            ]

            # ==================================
            # TRUNK LEAN
            # ==================================

            trunk_lean_angle = angle_from_vertical(
                shoulder_mid,
                hip_mid
            )

            # ==================================
            # ASYMMETRY
            # ==================================

            shoulder_height_difference = abs(
                left_shoulder[1] - right_shoulder[1]
            )

            hip_height_difference = abs(
                left_hip[1] - right_hip[1]
            )

            shoulder_hip_asymmetry = (
                shoulder_height_difference +
                hip_height_difference
            )

            # ==================================
            # TWIST FEATURES
            # ==================================

            shoulder_line_angle = line_angle(
                left_shoulder,
                right_shoulder
            )

            hip_line_angle = line_angle(
                left_hip,
                right_hip
            )

            twist_angle = abs(
                shoulder_line_angle - hip_line_angle
            )

            # ==================================
            # ARM FEATURES
            # ==================================

            left_arm_angle = calculate_angle(
                left_shoulder,
                left_elbow,
                left_wrist
            )

            right_arm_angle = calculate_angle(
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # ==================================
            # CREATE 12 FEATURES
            # ==================================

            feature_columns = [
                "left_knee_angle",
                "right_knee_angle",
                "average_knee_angle",
                "left_hip_angle",
                "right_hip_angle",
                "trunk_lean_angle",
                "shoulder_hip_asymmetry",
                "shoulder_line_angle",
                "hip_line_angle",
                "twist_angle",
                "left_arm_angle",
                "right_arm_angle"
            ]

            feature_values = [[
                left_knee_angle,
                right_knee_angle,
                average_knee_angle,
                left_hip_angle,
                right_hip_angle,
                trunk_lean_angle,
                shoulder_hip_asymmetry,
                shoulder_line_angle,
                hip_line_angle,
                twist_angle,
                left_arm_angle,
                right_arm_angle
            ]]

            features = pd.DataFrame(
                feature_values,
                columns=feature_columns
            )

            # ==================================
            # MODEL PREDICTION
            # ==================================

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            confidence = max(probabilities) * 100

            label = label_mapping.get(
                int(prediction),
                "UNKNOWN"
            )

            if label == "SAFE":
                display_text = "SAFE POSTURE"
                display_color = (0, 255, 0)
            else:
                display_text = (
                    "WARNING: " +
                    label.replace("_", " ")
                )
                display_color = (0, 0, 255)

            # ==================================
            # DRAW SKELETON
            # ==================================

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # ==================================
            # DISPLAY RESULTS
            # ==================================

            cv2.putText(
                frame,
                display_text,
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                display_color,
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Trunk Lean: {trunk_lean_angle:.1f}",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Average Knee: {average_knee_angle:.1f}",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "ErgoGuard - Live Posture Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ==========================================
# CLOSE WEBCAM
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print("Webcam stopped.")
print("========================================")