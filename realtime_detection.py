import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import pandas as pd


# ============================================================
# ERGOGUARD - REAL-TIME POSTURE DETECTION
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "models/ergoguard_model.pkl"
LABEL_FILE = "models/label_mapping.json"


# ============================================================
# LOAD MODEL AND LABEL MAPPING
# ============================================================

if not os.path.exists(MODEL_FILE):
    print("❌ Model file not found:", MODEL_FILE)
    exit()

if not os.path.exists(LABEL_FILE):
    print("❌ Label mapping file not found:", LABEL_FILE)
    exit()

model = joblib.load(MODEL_FILE)

with open(LABEL_FILE, "r") as f:
    label_mapping = json.load(f)

print("✅ Model loaded successfully")
print("✅ Label mapping loaded successfully")


# ============================================================
# FEATURE NAMES
# MUST MATCH TRAINING ORDER EXACTLY
# ============================================================

feature_names = [
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


# ============================================================
# ANGLE CALCULATION FUNCTION
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return np.nan

    cosine_angle = np.dot(ba, bc) / denominator

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cosine_angle))


# ============================================================
# GET LANDMARK POINT
# ============================================================

def get_point(landmarks, index):

    return [
        landmarks[index].x,
        landmarks[index].y
    ]


# ============================================================
# GET LABEL NAME
# ============================================================

def get_label_name(prediction):

    # Case 1: mapping is like {"0": "SAFE", "1": "SIDE_BENDING"}
    if isinstance(label_mapping, dict):

        prediction_string = str(int(prediction))

        if prediction_string in label_mapping:
            return label_mapping[prediction_string]

        if prediction in label_mapping:
            return label_mapping[prediction]

        # Case 2: mapping may contain label -> number
        for key, value in label_mapping.items():

            try:
                if int(value) == int(prediction):
                    return key
            except:
                pass

    return str(prediction)


# ============================================================
# CALCULATE ALL 12 FEATURES
# ============================================================

def calculate_features(landmarks):

    # --------------------------------------------------------
    # BODY LANDMARKS
    # MediaPipe Pose landmark indexes
    # --------------------------------------------------------

    left_shoulder = get_point(landmarks, 11)
    right_shoulder = get_point(landmarks, 12)

    left_elbow = get_point(landmarks, 13)
    right_elbow = get_point(landmarks, 14)

    left_wrist = get_point(landmarks, 15)
    right_wrist = get_point(landmarks, 16)

    left_hip = get_point(landmarks, 23)
    right_hip = get_point(landmarks, 24)

    left_knee = get_point(landmarks, 25)
    right_knee = get_point(landmarks, 26)

    left_ankle = get_point(landmarks, 27)
    right_ankle = get_point(landmarks, 28)


    # --------------------------------------------------------
    # LEFT KNEE ANGLE
    # --------------------------------------------------------

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )


    # --------------------------------------------------------
    # RIGHT KNEE ANGLE
    # --------------------------------------------------------

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )


    # --------------------------------------------------------
    # AVERAGE KNEE ANGLE
    # --------------------------------------------------------

    average_knee_angle = (
        left_knee_angle + right_knee_angle
    ) / 2


    # --------------------------------------------------------
    # LEFT HIP ANGLE
    # --------------------------------------------------------

    left_hip_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_knee
    )


    # --------------------------------------------------------
    # RIGHT HIP ANGLE
    # --------------------------------------------------------

    right_hip_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_knee
    )


    # --------------------------------------------------------
    # MIDPOINTS
    # --------------------------------------------------------

    shoulder_mid = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ]

    hip_mid = [
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ]


    # --------------------------------------------------------
    # TRUNK LEAN ANGLE
    # Angle between torso and vertical
    # --------------------------------------------------------

    vertical_reference = [
        hip_mid[0],
        hip_mid[1] - 1.0
    ]

    trunk_lean_angle = calculate_angle(
        shoulder_mid,
        hip_mid,
        vertical_reference
    )


    # --------------------------------------------------------
    # SHOULDER-HIP ASYMMETRY
    # --------------------------------------------------------

    shoulder_difference = abs(
        left_shoulder[1] - right_shoulder[1]
    )

    hip_difference = abs(
        left_hip[1] - right_hip[1]
    )

    shoulder_hip_asymmetry = (
        shoulder_difference + hip_difference
    )


    # --------------------------------------------------------
    # SHOULDER LINE ANGLE
    # --------------------------------------------------------

    horizontal_reference = [
        shoulder_mid[0] + 1.0,
        shoulder_mid[1]
    ]

    shoulder_line_angle = calculate_angle(
        left_shoulder,
        shoulder_mid,
        horizontal_reference
    )


    # --------------------------------------------------------
    # HIP LINE ANGLE
    # --------------------------------------------------------

    hip_line_angle = calculate_angle(
        left_hip,
        hip_mid,
        [hip_mid[0] + 1.0, hip_mid[1]]
    )


    # --------------------------------------------------------
    # TWIST ANGLE
    # --------------------------------------------------------

    twist_angle = abs(
        shoulder_line_angle - hip_line_angle
    )


    # --------------------------------------------------------
    # LEFT ARM ANGLE
    # --------------------------------------------------------

    left_arm_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )


    # --------------------------------------------------------
    # RIGHT ARM ANGLE
    # --------------------------------------------------------

    right_arm_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )


    # --------------------------------------------------------
    # RETURN FEATURES IN TRAINING ORDER
    # --------------------------------------------------------

    features = [
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
    ]

    return features


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not open camera.")
    exit()

print()
print("==============================================")
print("ERGOGUARD REAL-TIME DETECTION")
print("==============================================")
print("Camera started successfully.")
print("Show your full body to the camera.")
print("Press Q to quit.")
print()


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Could not read camera frame.")
        break


    # --------------------------------------------------------
    # FLIP CAMERA FOR MIRROR VIEW
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # CONVERT BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE POSE DETECTION
    # --------------------------------------------------------

    results = pose.process(rgb_frame)


    # --------------------------------------------------------
    # IF PERSON IS DETECTED
    # --------------------------------------------------------

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark


        # ----------------------------------------------------
        # DRAW BODY SKELETON
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # ----------------------------------------------------
        # CALCULATE FEATURES
        # ----------------------------------------------------

        features = calculate_features(landmarks)
        # Get trunk lean angle from calculated features
        # Get individual feature values for display
        left_knee_angle = features[feature_names.index("left_knee_angle")]
        right_knee_angle = features[feature_names.index("right_knee_angle")]
        average_knee_angle = features[feature_names.index("average_knee_angle")]

        left_hip_angle = features[feature_names.index("left_hip_angle")]
        right_hip_angle = features[feature_names.index("right_hip_angle")]

        trunk_lean_angle = features[feature_names.index("trunk_lean_angle")]

        shoulder_hip_asymmetry = features[feature_names.index("shoulder_hip_asymmetry")]
        shoulder_line_angle = features[feature_names.index("shoulder_line_angle")]
        hip_line_angle = features[feature_names.index("hip_line_angle")]
        twist_angle = features[feature_names.index("twist_angle")]

        left_arm_angle = features[feature_names.index("left_arm_angle")]
        right_arm_angle = features[feature_names.index("right_arm_angle")]
        

        # ----------------------------------------------------
        # CHECK FOR INVALID VALUES
        # ----------------------------------------------------

        if not any(np.isnan(features)):

            # ------------------------------------------------
            # IMPORTANT:
            # USE DATAFRAME WITH FEATURE NAMES
            # ------------------------------------------------

            features_df = pd.DataFrame(
                [features],
                columns=feature_names
            )


            # ------------------------------------------------
            # ML PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                features_df
            )[0]


            # ------------------------------------------------
            # GET POSTURE LABEL
            # ------------------------------------------------

            label = get_label_name(prediction)


            # ------------------------------------------------
            # DISPLAY TEXT
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (10, 10),
                (600, 100),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                f"POSTURE: {label}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Trunk: {trunk_lean_angle:.1f} deg",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # UNSAFE POSTURE MESSAGE
            # ------------------------------------------------

            unsafe_labels = [
                "EXCESSIVE_FORWARD_BENDING",
                "EXCESSIVE_SQUATTING",
                "SIDE_BENDING",
                "TWISTED_TRUNK",
                "ASYMMETRICAL_LIFTING"
            ]


            if label in unsafe_labels:

                cv2.putText(
                    frame,
                    "WARNING: UNSAFE POSTURE",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )


            else:

                cv2.putText(
                    frame,
                    "STATUS: SAFE",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # DISPLAY ANGLES
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Left Knee: {left_knee_angle:.1f}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Knee: {right_knee_angle:.1f}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Twist: {twist_angle:.1f}",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    else:

        # ----------------------------------------------------
        # NO PERSON DETECTED
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 65),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # SHOW FRAME
    # --------------------------------------------------------

    cv2.imshow(
        "ErgoGuard - Real-Time Posture Detection",
        frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
pose.close()
cv2.destroyAllWindows()

print()
print("==============================================")
print("ERGOGUARD REAL-TIME DETECTION STOPPED")
print("==============================================")    