import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_pose_on_frame(frame_bgr, results):
    """
    Draw pose landmarks on a single BGR frame.
    Returns a BGR frame with overlay.
    """
    overlay = frame_bgr.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            overlay,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return overlay


def generate_pose_overlay_frames(sampled_frames_bgr, max_frames=4):
    """
    Run pose estimation on sampled frames and return a small list
    of overlay images for interface display.

    Parameters
    ----------
    sampled_frames_bgr : list[np.ndarray]
        Sampled video frames in BGR format.
    max_frames : int
        Maximum number of overlay frames to return.

    Returns
    -------
    overlay_frames_rgb : list[np.ndarray]
        Pose-overlay frames in RGB format for Streamlit display.
    """
    if len(sampled_frames_bgr) == 0:
        return []

    selected_indices = np.linspace(
        0, len(sampled_frames_bgr) - 1, num=min(max_frames, len(sampled_frames_bgr)), dtype=int
    )

    overlay_frames_rgb = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        for idx in selected_indices:
            frame_bgr = sampled_frames_bgr[idx]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = pose.process(frame_rgb)
            overlay_bgr = draw_pose_on_frame(frame_bgr, results)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            overlay_frames_rgb.append(overlay_rgb)

    return overlay_frames_rgb