import numpy as np

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)

def get_xy(frame_landmarks, idx):
    x = frame_landmarks[idx * 4]
    y = frame_landmarks[idx * 4 + 1]
    return np.array([x, y], dtype=np.float32)

def knee_angle(frame_landmarks, side="left"):
    if side == "left":
        hip = get_xy(frame_landmarks, LEFT_HIP)
        knee = get_xy(frame_landmarks, LEFT_KNEE)
        ankle = get_xy(frame_landmarks, LEFT_ANKLE)
    else:
        hip = get_xy(frame_landmarks, RIGHT_HIP)
        knee = get_xy(frame_landmarks, RIGHT_KNEE)
        ankle = get_xy(frame_landmarks, RIGHT_ANKLE)

    return compute_angle(hip, knee, ankle)

def summarize_biomechanics(sequence):
    mean_knee_angles = []

    for frame in sequence:
        left = knee_angle(frame, "left")
        right = knee_angle(frame, "right")
        mean_knee_angles.append((left + right) / 2.0)

    landing_idx = int(np.argmin(mean_knee_angles))
    landing_frame = sequence[landing_idx]

    left_knee = knee_angle(landing_frame, "left")
    right_knee = knee_angle(landing_frame, "right")

    return {
        "landing_frame_index": landing_idx,
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "mean_knee_angle": (left_knee + right_knee) / 2.0,
        "knee_symmetry_diff": abs(left_knee - right_knee),
    }