import pandas as pd
import numpy as np

# Read the landmark CSV
df = pd.read_csv("results_landmarks.csv")


# Function to calculate angle between 3 points
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

    angle = np.degrees(np.arccos(cosine_angle))

    return angle


# Lists to store calculated angles
left_knee_angles = []
right_knee_angles = []
trunk_angles = []


# Process every frame
for _, row in df.iterrows():

    # =========================
    # LEFT LEG
    # =========================

    left_hip = [
        row["landmark_23_x"],
        row["landmark_23_y"]
    ]

    left_knee = [
        row["landmark_25_x"],
        row["landmark_25_y"]
    ]

    left_ankle = [
        row["landmark_27_x"],
        row["landmark_27_y"]
    ]


    # =========================
    # RIGHT LEG
    # =========================

    right_hip = [
        row["landmark_24_x"],
        row["landmark_24_y"]
    ]

    right_knee = [
        row["landmark_26_x"],
        row["landmark_26_y"]
    ]

    right_ankle = [
        row["landmark_28_x"],
        row["landmark_28_y"]
    ]


    # =========================
    # KNEE ANGLES
    # =========================

    left_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    left_knee_angles.append(left_angle)
    right_knee_angles.append(right_angle)


    # =========================
    # TRUNK / BACK ANGLE
    # =========================

    # Midpoint between left and right shoulders
    shoulder_mid = [
        (
            row["landmark_11_x"] +
            row["landmark_12_x"]
        ) / 2,

        (
            row["landmark_11_y"] +
            row["landmark_12_y"]
        ) / 2
    ]


    # Midpoint between left and right hips
    hip_mid = [
        (
            row["landmark_23_x"] +
            row["landmark_24_x"]
        ) / 2,

        (
            row["landmark_23_y"] +
            row["landmark_24_y"]
        ) / 2
    ]


    # Vertical reference point above the hip
    vertical_point = [
        hip_mid[0],
        hip_mid[1] - 1
    ]


    trunk_angle = calculate_angle(
        shoulder_mid,
        hip_mid,
        vertical_point
    )

    trunk_angles.append(trunk_angle)


# =========================
# ADD ANGLES TO DATAFRAME
# =========================

df["left_knee_angle"] = left_knee_angles
df["right_knee_angle"] = right_knee_angles
df["trunk_angle"] = trunk_angles


# =========================
# SAVE NEW CSV
# =========================

df.to_csv(
    "results_with_angles.csv",
    index=False
)


# =========================
# DISPLAY RESULTS
# =========================

print("✅ Angle calculation completed!")
print("CSV saved as: results_with_angles.csv")

print("\nFirst 5 rows:")

print(
    df[
        [
            "frame",
            "left_knee_angle",
            "right_knee_angle",
            "trunk_angle"
        ]
    ].head()
)