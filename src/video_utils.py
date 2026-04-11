from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import CLIP_METADATA_PATH, PROJECT_ROOT


def load_clip_metadata(csv_path: Path = CLIP_METADATA_PATH) -> pd.DataFrame:
    """
    Load clip metadata from CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the clip metadata CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame containing clip_id, video_path, and label.
    """
    df = pd.read_csv(csv_path)

    required_columns = {"clip_id", "video_path", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata CSV: {missing_columns}")

    return df


def resolve_video_path(relative_video_path: str) -> Path:
    """
    Convert a relative video path from the CSV into an absolute project path.

    Parameters
    ----------
    relative_video_path : str
        Relative path such as 'data/raw/bad/bad_jump_01.MOV'.

    Returns
    -------
    Path
        Absolute path to the video file.
    """
    return PROJECT_ROOT / relative_video_path


def validate_video_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add path-validation columns to the metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clip metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with absolute_path and file_exists columns added.
    """
    df = df.copy()
    df["absolute_path"] = df["video_path"].apply(resolve_video_path)
    df["file_exists"] = df["absolute_path"].apply(lambda p: p.exists())
    return df


def get_video_frame_count(video_path: Path) -> int:
    """
    Return the total frame count of a video using OpenCV.

    Parameters
    ----------
    video_path : Path
        Absolute path to the video file.

    Returns
    -------
    int
        Total number of frames.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If OpenCV cannot open the video.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def sample_frame_indices(total_frames: int, num_samples: int = 10) -> np.ndarray:
    """
    Uniformly sample frame indices across a video.

    Parameters
    ----------
    total_frames : int
        Total number of frames in the video.
    num_samples : int
        Number of indices to sample.

    Returns
    -------
    np.ndarray
        Array of frame indices.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    return np.linspace(0, total_frames - 1, num=num_samples, dtype=int)


def load_sampled_frames(
    video_path: Path,
    num_samples: int = 10,
    convert_bgr_to_rgb: bool = True
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load uniformly sampled frames from a video.

    Parameters
    ----------
    video_path : Path
        Absolute path to the video file.
    num_samples : int
        Number of frames to sample.
    convert_bgr_to_rgb : bool
        Whether to convert frames from BGR to RGB.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        List of frames and the sampled frame indices.
    """
    total_frames = get_video_frame_count(video_path)
    frame_indices = sample_frame_indices(total_frames, num_samples=num_samples)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()

        if not success or frame is None:
            raise ValueError(f"Failed to read frame {idx} from {video_path}")

        if convert_bgr_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()
    return frames, frame_indices