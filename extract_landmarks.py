import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path


# ==========================================
# 1. DATASET LOCATION
# ==========================================

dataset_folder = Path("DATASET")

# Find all MP4 videos inside DATASET
video_files = list(dataset_folder.rglob("*.mp4"))

print(f"Found {len(video_files)} videos.")


# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

mp_pose = mp.solutions.pose

all_landmarks = []


# ==========================================
# 3. PROCESS EACH VIDEO
# ==========================================

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    for video_number, video_path in enumerate(video_files, start=1):

        print(
            f"\nProcessing video {video_number}/{len(video_files)}: "
            f"{video_path.name}"
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("❌ Could not open:", video_path)
            continue

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Convert BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Detect pose
            results = pose.process(rgb_frame)

            # Save landmarks if detected
            if results.pose_landmarks:

                row = {
                    "video": video_path.name,
                    "frame": frame_number
                }

                # MediaPipe has 33 body landmarks
                for landmark_id, landmark in enumerate(
                    results.pose_landmarks.landmark
                ):

                    row[f"landmark_{landmark_id}_x"] = landmark.x
                    row[f"landmark_{landmark_id}_y"] = landmark.y
                    row[f"landmark_{landmark_id}_z"] = landmark.z
                    row[f"landmark_{landmark_id}_visibility"] = (
                        landmark.visibility
                    )

                all_landmarks.append(row)

        cap.release()

        print(
            f"   Frames processed: {frame_number}"
        )


# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(all_landmarks)


# ==========================================
# 5. SAVE DATASET
# ==========================================

output_file = "all_landmarks.csv"

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 6. SUMMARY
# ==========================================

print("\n===================================")
print("✅ LANDMARK EXTRACTION COMPLETED!")
print("===================================")

print(f"Videos found: {len(video_files)}")
print(f"Landmark rows: {len(df)}")
print(f"CSV saved as: {output_file}")

print("\nColumns:")
print(df.columns.tolist()[:10], "...")