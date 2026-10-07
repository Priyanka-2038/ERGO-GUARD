import pandas as pd
import numpy as np

# ==========================================
# 1. READ THE LANDMARK DATASET
# ==========================================

df = pd.read_csv("all_landmarks.csv")

print(f"Loaded {len(df)} landmark rows.")


# ==========================================
# 2. FUNCTION TO CALCULATE ANGLE
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


# ==========================================
# 3. CALCULATE ANGLES
# ==========================================

left_knee_angles = []
right_knee_angles = []
trunk_angles = []


for _, row in df.iterrows():

    # --------------------------------------
    # LEFT LEG
    # --------------------------------------

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


    # --------------------------------------
    # RIGHT LEG
    # --------------------------------------

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


    # --------------------------------------
    # KNEE ANGLES
    # --------------------------------------

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


    # --------------------------------------
    # TRUNK ANGLE
    # --------------------------------------

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


    # Vertical reference point
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


# ==========================================
# 4. ADD ANGLES TO DATASET
# ==========================================

df["left_knee_angle"] = left_knee_angles
df["right_knee_angle"] = right_knee_angles
df["trunk_angle"] = trunk_angles


# ==========================================
# 5. SAVE FINAL ANGLE DATASET
# ==========================================

output_file = "all_angles.csv"

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("✅ ANGLE CALCULATION COMPLETED!")
print("===================================")

print(f"Total rows: {len(df)}")
print(f"CSV saved as: {output_file}")

print("\nFirst 5 rows:")

print(
    df[
        [
            "video",
            "frame",
            "left_knee_angle",
            "right_knee_angle",
            "trunk_angle"
        ]
    ].head()
)