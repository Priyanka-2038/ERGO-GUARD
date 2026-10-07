# ============================================================
# ERGOGUARD - POSTURE CORRECTION & SAFE COUNTERPARTS
# ============================================================

POSTURE_CORRECTIONS = {

    "EXCESSIVE_FORWARD_BENDING": {
        "status": "UNSAFE",
        "safe_posture": "CORRECT FORWARD-BENDING / LIFTING POSTURE",
        "message": "Reduce excessive forward bending.",
        "correction": [
            "Keep your back in a more neutral position.",
            "Bend through your hips and knees appropriately.",
            "Keep the load closer to your body."
        ]
    },

    "SIDE_BENDING": {
        "status": "UNSAFE",
        "safe_posture": "BALANCED UPRIGHT POSTURE",
        "message": "Avoid excessive sideways bending.",
        "correction": [
            "Keep your torso more centered.",
            "Keep your shoulders and hips aligned.",
            "Move your feet instead of leaning excessively."
        ]
    },

    "TWISTED_TRUNK": {
        "status": "UNSAFE",
        "safe_posture": "NON-TWISTED LIFTING POSTURE",
        "message": "Avoid twisting your trunk while handling the load.",
        "correction": [
            "Face the load directly.",
            "Keep your shoulders and hips aligned.",
            "Turn your whole body instead of twisting your trunk."
        ]
    },

    "EXCESSIVE_SQUATTING": {
        "status": "UNSAFE",
        "safe_posture": "CONTROLLED SQUAT / LIFTING POSTURE",
        "message": "Avoid excessive squatting.",
        "correction": [
            "Use a controlled knee and hip bend.",
            "Keep your back in a neutral position.",
            "Keep the load close to your body."
        ]
    },

    "OVERHEAD_OR_REACHING": {
        "status": "UNSAFE",
        "safe_posture": "SAFE REACHING POSTURE",
        "message": "Avoid excessive overhead reaching.",
        "correction": [
            "Keep the load within a comfortable reach.",
            "Avoid unnecessary overhead reaching.",
            "Reposition yourself when possible."
        ]
    },

    "BENDING_WITH_STRAIGHT_KNEES": {
        "status": "UNSAFE",
        "safe_posture": "KNEE-BENT LIFTING POSTURE",
        "message": "Avoid bending with straight knees.",
        "correction": [
            "Bend your knees when lowering toward the load.",
            "Use your hips and knees during the movement.",
            "Keep your back in a neutral position."
        ]
    },

    "ASYMMETRICAL_LIFTING": {
        "status": "UNSAFE",
        "safe_posture": "SYMMETRICAL LIFTING POSTURE",
        "message": "Avoid an uneven lifting position.",
        "correction": [
            "Use a balanced stance.",
            "Keep the load centered.",
            "Keep your body aligned with the load."
        ]
    },

    "ROUNDED_BACK": {
        "status": "UNSAFE",
        "safe_posture": "NEUTRAL-BACK POSTURE",
        "message": "Avoid excessive rounding of your back.",
        "correction": [
            "Maintain a more neutral back position.",
            "Use your hips and knees during lifting.",
            "Keep the load close to your body."
        ]
    },

    "SAFE": {
        "status": "SAFE",
        "safe_posture": "CURRENT POSTURE",
        "message": "Good posture detected.",
        "correction": [
            "Continue maintaining your current posture."
        ]
    }
}


def get_posture_correction(posture):
    """
    Return the safe counterpart and correction instructions
    for a detected posture.
    """

    posture = str(posture).strip().upper()

    return POSTURE_CORRECTIONS.get(
        posture,
        {
            "status": "UNKNOWN",
            "safe_posture": "UNKNOWN",
            "message": "Posture not recognized.",
            "correction": []
        }
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_posture = "EXCESSIVE_FORWARD_BENDING"

    result = get_posture_correction(test_posture)

    print("\n==============================================")
    print("           ERGOGUARD CORRECTION")
    print("==============================================")

    print("Detected posture :", test_posture)
    print("Status            :", result["status"])
    print("Safe counterpart  :", result["safe_posture"])
    print("Message           :", result["message"])

    print("\nCorrection:")
    for instruction in result["correction"]:
        print(" -", instruction)

    print("==============================================")