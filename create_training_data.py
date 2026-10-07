import pandas as pd
import numpy as np

# ==========================================
# ERGOGUARD - CREATE MULTI-CLASS TRAINING DATA
# ==========================================

input_file = "all_landmarks.csv"
output_file = "training_data.csv"

print("==========================================")
print("ERGOGUARD - MULTI-CLASS TRAINING DATA")
print("==========================================")

df = pd.read_csv(input_file)

print("Rows loaded:", len(df))


# ==========================================
# ANGLE FUNCTION
# ==========================================

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


def angle_from_vertical(shoulder_mid, hip_mid):

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    # 0 degrees means upright.
    # Larger values mean more trunk leaning.
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def line_angle(point_a, point_b):

    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]

    return np.degrees(np.arctan2(dy, dx))


# ==========================================
# FEATURE LISTS
# ==========================================

left_knee_angles = []
right_knee_angles = []

left_hip_angles = []
right_hip_angles = []

average_knee_angles = []

trunk_lean_angles = []
shoulder_hip_asymmetry = []

shoulder_line_angles = []
hip_line_angles = []
twist_angles = []

left_arm_angles = []
right_arm_angles = []

labels = []


# ==========================================
# PROCESS EVERY FRAME
# ==========================================

for _, row in df.iterrows():

    # --------------------------------------
    # BODY LANDMARKS
    # --------------------------------------

    left_shoulder = [
        row["landmark_11_x"],
        row["landmark_11_y"]
    ]

    right_shoulder = [
        row["landmark_12_x"],
        row["landmark_12_y"]
    ]

    left_hip = [
        row["landmark_23_x"],
        row["landmark_23_y"]
    ]

    right_hip = [
        row["landmark_24_x"],
        row["landmark_24_y"]
    ]

    left_knee = [
        row["landmark_25_x"],
        row["landmark_25_y"]
    ]

    right_knee = [
        row["landmark_26_x"],
        row["landmark_26_y"]
    ]

    left_ankle = [
        row["landmark_27_x"],
        row["landmark_27_y"]
    ]

    right_ankle = [
        row["landmark_28_x"],
        row["landmark_28_y"]
    ]

    left_elbow = [
        row["landmark_13_x"],
        row["landmark_13_y"]
    ]

    right_elbow = [
        row["landmark_14_x"],
        row["landmark_14_y"]
    ]

    left_wrist = [
        row["landmark_15_x"],
        row["landmark_15_y"]
    ]

    right_wrist = [
        row["landmark_16_x"],
        row["landmark_16_y"]
    ]

    # --------------------------------------
    # KNEE ANGLES
    # --------------------------------------

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
        left_knee_angle + right_knee_angle
    ) / 2

    # --------------------------------------
    # HIP ANGLES
    # --------------------------------------

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

    # --------------------------------------
    # MIDPOINTS
    # --------------------------------------

    shoulder_mid = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ]

    hip_mid = [
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ]

    # --------------------------------------
    # TRUNK LEAN
    # --------------------------------------

    trunk_lean_angle = angle_from_vertical(
        shoulder_mid,
        hip_mid
    )

    # --------------------------------------
    # SHOULDER AND HIP ASYMMETRY
    # --------------------------------------

    shoulder_height_difference = abs(
        left_shoulder[1] - right_shoulder[1]
    )

    hip_height_difference = abs(
        left_hip[1] - right_hip[1]
    )

    asymmetry = (
        shoulder_height_difference +
        hip_height_difference
    )

    # --------------------------------------
    # TRUNK TWIST PROXY
    # --------------------------------------

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

    # --------------------------------------
    # ARM ANGLES
    # --------------------------------------

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

    # ======================================
    # SAVE FEATURES
    # ======================================

    left_knee_angles.append(left_knee_angle)
    right_knee_angles.append(right_knee_angle)

    left_hip_angles.append(left_hip_angle)
    right_hip_angles.append(right_hip_angle)

    average_knee_angles.append(average_knee_angle)

    trunk_lean_angles.append(trunk_lean_angle)
    shoulder_hip_asymmetry.append(asymmetry)

    shoulder_line_angles.append(shoulder_line_angle)
    hip_line_angles.append(hip_line_angle)
    twist_angles.append(twist_angle)

    left_arm_angles.append(left_arm_angle)
    right_arm_angles.append(right_arm_angle)

    # ======================================
    # MULTI-CLASS LABELING
    # ======================================

    label = "SAFE"

    if trunk_lean_angle > 45 and average_knee_angle > 160:
        label = "BENDING_WITH_STRAIGHT_KNEES"

    elif trunk_lean_angle > 45:
        label = "EXCESSIVE_FORWARD_BENDING"

    elif average_knee_angle < 70:
        label = "EXCESSIVE_SQUATTING"

    elif twist_angle > 25:
        label = "TWISTED_TRUNK"

    elif asymmetry > 0.08:
        label = "ASYMMETRICAL_LIFTING"

    elif left_arm_angle < 70 or right_arm_angle < 70:
        label = "OVERHEAD_OR_REACHING"

    elif trunk_lean_angle > 25:
        label = "SIDE_BENDING"

    elif left_hip_angle < 70 or right_hip_angle < 70:
        label = "ROUNDED_BACK"

    labels.append(label)


# ==========================================
# ADD FEATURES
# ==========================================

df["left_knee_angle"] = left_knee_angles
df["right_knee_angle"] = right_knee_angles
df["average_knee_angle"] = average_knee_angles

df["left_hip_angle"] = left_hip_angles
df["right_hip_angle"] = right_hip_angles

df["trunk_lean_angle"] = trunk_lean_angles
df["shoulder_hip_asymmetry"] = shoulder_hip_asymmetry

df["shoulder_line_angle"] = shoulder_line_angles
df["hip_line_angle"] = hip_line_angles
df["twist_angle"] = twist_angles

df["left_arm_angle"] = left_arm_angles
df["right_arm_angle"] = right_arm_angles

df["label"] = labels


# ==========================================
# NUMERIC LABELS
# ==========================================

label_mapping = {
    "SAFE": 0,
    "EXCESSIVE_FORWARD_BENDING": 1,
    "BENDING_WITH_STRAIGHT_KNEES": 2,
    "ROUNDED_BACK": 3,
    "TWISTED_TRUNK": 4,
    "SIDE_BENDING": 5,
    "ASYMMETRICAL_LIFTING": 6,
    "EXCESSIVE_SQUATTING": 7,
    "OVERHEAD_OR_REACHING": 8
}

df["label_numeric"] = df["label"].map(label_mapping)


# ==========================================
# REMOVE INVALID ROWS
# ==========================================

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

df = df.dropna(
    subset=feature_columns
).copy()


# ==========================================
# SAVE DATA
# ==========================================

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("MULTI-CLASS TRAINING DATA CREATED")
print("==========================================")

print("Rows used:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFeatures used:")

for feature in feature_columns:
    print("-", feature)

print("\nOutput file:", output_file)

print("\n==========================================")
print("DONE")
print("==========================================")