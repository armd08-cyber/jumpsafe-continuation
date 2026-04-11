from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLIP_METADATA_PATH,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
)
from src.video_utils import (
    load_clip_metadata,
    validate_video_paths,
    load_sampled_frames,
)
from src.pose_extraction import (
    initialize_pose_estimator,
    extract_pose_sequence,
    count_missing_pose_frames,
)
from src.preprocessing import preprocess_landmark_sequence


LANDMARK_SEQUENCE_DIR = PROCESSED_DIR / "landmark_sequences"
PROCESSED_METADATA_PATH = PROCESSED_DIR / "processed_sequence_metadata.csv"


def ensure_output_directories() -> None:
    """
    Create output directories if they do not already exist.
    """
    LANDMARK_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)


def process_single_clip(
    clip_id: str,
    video_path: Path,
    label: str,
    pose_estimator,
    target_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Process one video clip into a fixed-length landmark sequence and save it.

    Parameters
    ----------
    clip_id : str
        Clip identifier.
    video_path : Path
        Absolute path to the raw video.
    label : str
        Clip label.
    pose_estimator :
        Initialized MediaPipe Pose estimator.
    target_length : int
        Desired sequence length.

    Returns
    -------
    dict
        Metadata about the processed clip.
    """
    frames, frame_indices = load_sampled_frames(
        video_path=video_path,
        num_samples=target_length,
        convert_bgr_to_rgb=True,
    )

    raw_landmarks, pose_results = extract_pose_sequence(frames, pose_estimator)
    missing_pose_frames = count_missing_pose_frames(raw_landmarks)

    processed_sequence = preprocess_landmark_sequence(
        raw_landmarks,
        target_length=target_length,
    )

    output_path = LANDMARK_SEQUENCE_DIR / f"{clip_id}.npy"
    np.save(output_path, processed_sequence)

    return {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "label": label,
        "sequence_path": str(output_path),
        "num_sampled_frames": len(frame_indices),
        "missing_pose_frames": missing_pose_frames,
        "processed_shape": str(processed_sequence.shape),
    }


def build_all_sequences(
    metadata_csv_path: Path = CLIP_METADATA_PATH,
    target_length: int = SEQUENCE_LENGTH,
) -> pd.DataFrame:
    """
    Process all clips listed in the metadata CSV and save processed sequences.

    Parameters
    ----------
    metadata_csv_path : Path
        Path to clip metadata CSV.
    target_length : int
        Desired sequence length.

    Returns
    -------
    pd.DataFrame
        Processed metadata DataFrame.
    """
    ensure_output_directories()

    df = load_clip_metadata(metadata_csv_path)
    df = validate_video_paths(df)

    if not df["file_exists"].all():
        missing = df.loc[~df["file_exists"], ["clip_id", "video_path", "absolute_path"]]
        raise FileNotFoundError(
            f"Some video files are missing:\n{missing.to_string(index=False)}"
        )

    pose_estimator = initialize_pose_estimator()

    records = []
    try:
        for _, row in df.iterrows():
            clip_id = row["clip_id"]
            video_path = row["absolute_path"]
            label = row["label"]

            record = process_single_clip(
                clip_id=clip_id,
                video_path=video_path,
                label=label,
                pose_estimator=pose_estimator,
                target_length=target_length,
            )
            records.append(record)
            print(f"Processed: {clip_id}")
    finally:
        pose_estimator.close()

    processed_df = pd.DataFrame(records)
    processed_df.to_csv(PROCESSED_METADATA_PATH, index=False)

    return processed_df