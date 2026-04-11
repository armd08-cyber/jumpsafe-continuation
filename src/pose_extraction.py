from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def initialize_pose_estimator(
    static_image_mode: bool = False,
    model_complexity: int = 1,
    enable_segmentation: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """
    Initialize and return a MediaPipe Pose estimator.
    """
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return pose


def extract_landmarks_from_result(result) -> np.ndarray:
    """
    Extract pose landmarks from a MediaPipe result.

    Returns
    -------
    np.ndarray
        Array of shape (33, 4) with columns:
        [x, y, z, visibility]

        Returns zeros if no pose is detected.
    """
    if result.pose_landmarks is None:
        return np.zeros((33, 4), dtype=np.float32)

    landmarks = []
    for lm in result.pose_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(landmarks, dtype=np.float32)


def run_pose_on_frame(frame_rgb: np.ndarray, pose_estimator) -> Tuple[np.ndarray, object]:
    """
    Run MediaPipe Pose on one RGB frame.

    Parameters
    ----------
    frame_rgb : np.ndarray
        RGB frame of shape (H, W, 3).
    pose_estimator :
        Initialized MediaPipe Pose estimator.

    Returns
    -------
    landmarks : np.ndarray
        Array of shape (33, 4).
    result : object
        Raw MediaPipe result.
    """
    result = pose_estimator.process(frame_rgb)
    landmarks = extract_landmarks_from_result(result)
    return landmarks, result


def extract_pose_sequence(
    frames_rgb: List[np.ndarray],
    pose_estimator
) -> Tuple[np.ndarray, List[object]]:
    """
    Run pose extraction on a sequence of RGB frames.

    Parameters
    ----------
    frames_rgb : List[np.ndarray]
        List of sampled RGB frames.
    pose_estimator :
        Initialized MediaPipe Pose estimator.

    Returns
    -------
    landmark_sequence : np.ndarray
        Array of shape (T, 33, 4), where T is the number of frames.
    results : List[object]
        List of raw MediaPipe results for visualization/debugging.
    """
    all_landmarks = []
    all_results = []

    for frame in frames_rgb:
        landmarks, result = run_pose_on_frame(frame, pose_estimator)
        all_landmarks.append(landmarks)
        all_results.append(result)

    landmark_sequence = np.stack(all_landmarks, axis=0).astype(np.float32)
    return landmark_sequence, all_results


def draw_pose_on_frame(frame_rgb: np.ndarray, result) -> np.ndarray:
    """
    Draw pose landmarks on an RGB frame.

    Parameters
    ----------
    frame_rgb : np.ndarray
        Input RGB frame.
    result : object
        MediaPipe pose result.

    Returns
    -------
    np.ndarray
        Annotated RGB frame.
    """
    annotated_frame = frame_rgb.copy()

    if result.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated_frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

    return annotated_frame


def count_missing_pose_frames(landmark_sequence: np.ndarray) -> int:
    """
    Count frames where no pose was detected.

    Parameters
    ----------
    landmark_sequence : np.ndarray
        Array of shape (T, 33, 4).

    Returns
    -------
    int
        Number of frames that are entirely zero.
    """
    return int(np.sum(np.all(landmark_sequence == 0, axis=(1, 2))))