import numpy as np


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def clean_landmark_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Replace NaNs and infinities in a landmark sequence.

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Shape (T, 33, 4)

    Returns
    -------
    np.ndarray
        Cleaned landmark sequence.
    """
    cleaned = np.array(landmark_sequence, dtype=np.float32, copy=True)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned


def compute_midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """
    Compute midpoint between two 3D landmark coordinates.
    """
    return (point_a + point_b) / 2.0


def normalize_single_frame(frame_landmarks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize one frame by centering on hip midpoint and scaling by shoulder width.

    Parameters
    ----------
    frame_landmarks : np.ndarray
        Shape (33, 4) with columns [x, y, z, visibility]

    Returns
    -------
    np.ndarray
        Normalized frame of shape (33, 4)
    """
    normalized = frame_landmarks.copy().astype(np.float32)

    coords = normalized[:, :3]
    visibility = normalized[:, 3:]

    left_hip = coords[LEFT_HIP]
    right_hip = coords[RIGHT_HIP]
    hip_center = compute_midpoint(left_hip, right_hip)

    left_shoulder = coords[LEFT_SHOULDER]
    right_shoulder = coords[RIGHT_SHOULDER]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_width < eps:
        shoulder_width = 1.0

    coords = (coords - hip_center) / shoulder_width

    normalized[:, :3] = coords
    normalized[:, 3:] = visibility

    return normalized


def normalize_landmark_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Normalize all frames in a landmark sequence.

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Shape (T, 33, 4)

    Returns
    -------
    np.ndarray
        Shape (T, 33, 4)
    """
    normalized_frames = [normalize_single_frame(frame) for frame in landmark_sequence]
    return np.stack(normalized_frames, axis=0).astype(np.float32)


def pad_or_truncate_sequence(landmark_sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad with zeros or truncate to a fixed sequence length.

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Shape (T, 33, 4)
    target_length : int
        Desired sequence length

    Returns
    -------
    np.ndarray
        Shape (target_length, 33, 4)
    """
    current_length = landmark_sequence.shape[0]

    if current_length == target_length:
        return landmark_sequence

    if current_length > target_length:
        return landmark_sequence[:target_length]

    pad_length = target_length - current_length
    pad_shape = (pad_length, landmark_sequence.shape[1], landmark_sequence.shape[2])
    padding = np.zeros(pad_shape, dtype=landmark_sequence.dtype)

    return np.concatenate([landmark_sequence, padding], axis=0)


def flatten_landmark_sequence(landmark_sequence: np.ndarray) -> np.ndarray:
    """
    Flatten each frame from (33, 4) to (132,).

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Shape (T, 33, 4)

    Returns
    -------
    np.ndarray
        Shape (T, 132)
    """
    t, n_landmarks, n_features = landmark_sequence.shape
    return landmark_sequence.reshape(t, n_landmarks * n_features).astype(np.float32)


def preprocess_landmark_sequence(
    landmark_sequence: np.ndarray,
    target_length: int = 30
) -> np.ndarray:
    """
    Full preprocessing pipeline:
    clean -> normalize -> pad/truncate -> flatten

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Shape (T, 33, 4)
    target_length : int
        Desired fixed sequence length

    Returns
    -------
    np.ndarray
        Shape (target_length, 132)
    """
    cleaned = clean_landmark_sequence(landmark_sequence)
    normalized = normalize_landmark_sequence(cleaned)
    fixed_length = pad_or_truncate_sequence(normalized, target_length=target_length)
    flattened = flatten_landmark_sequence(fixed_length)
    return flattened