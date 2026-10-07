import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

print("\n========================================")
print("     ERGOGUARD - MULTI-CLASS TRAINING")
print("========================================\n")

# ==========================================
# LOAD DATA
# ==========================================

data_file = "training_data.csv"

if not os.path.exists(data_file):
    print("❌ training_data.csv not found!")
    exit()

df = pd.read_csv(data_file)

print("✅ Dataset loaded successfully!")
print(f"Total samples: {len(df)}")


# ==========================================
# FEATURES
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

for column in feature_columns:
    if column not in df.columns:
        print(f"❌ Missing column: {column}")
        exit()

if "label" not in df.columns:
    print("❌ Label column not found!")
    exit()


# ==========================================
# LABEL MAPPING
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

inverse_mapping = {
    value: key
    for key, value in label_mapping.items()
}


# ==========================================
# PREPARE DATA
# ==========================================

X = df[feature_columns].copy()
y = df["label"].astype(str).str.strip().str.upper()

unknown_labels = set(y.unique()) - set(label_mapping.keys())

if unknown_labels:
    print("❌ Unknown labels found:", unknown_labels)
    exit()

y = y.map(label_mapping)

valid_rows = X.notna().all(axis=1) & y.notna()

X = X[valid_rows]
y = y[valid_rows].astype(int)

print(f"\nValid samples: {len(X)}")

print("\nClass distribution:")

for label_name, label_number in label_mapping.items():
    print(
        f"{label_name}: {(y == label_number).sum()}"
    )


# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("DATA SPLIT")
print("========================================")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ==========================================
# CREATE MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n========================================")
print("TRAINING MODEL...")
print("========================================")

model.fit(X_train, y_train)

print("✅ Multi-class model training completed!")


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(label_mapping.values()),
        target_names=list(label_mapping.keys()),
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=list(label_mapping.values())
    )
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

for feature, importance in zip(
    feature_columns,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")


# ==========================================
# SAVE MODEL AND LABEL MAPPING
# ==========================================

os.makedirs("models", exist_ok=True)

model_file = "models/ergoguard_model.pkl"
mapping_file = "models/label_mapping.json"

joblib.dump(model, model_file)

with open(mapping_file, "w") as file:
    json.dump(inverse_mapping, file, indent=4)

print("\n========================================")
print("✅ MODEL SAVED SUCCESSFULLY!")
print("========================================")

print(f"Model file: {model_file}")
print(f"Label mapping file: {mapping_file}")

print("\n🚀 ErgoGuard multi-class training completed!")