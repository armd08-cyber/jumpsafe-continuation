from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np

from src.config import INDEX_TO_LABEL, SEQUENCE_LENGTH
from src.video_utils import load_sampled_frames
from src.pose_extraction import (
    initialize_pose_estimator,
    extract_pose_sequence,
    count_missing_pose_frames,
)
from src.preprocessing import preprocess_landmark_sequence
from src.train_baselines import BASELINE_MODEL_PATH


def aggregate_sequence_mean(sequence: np.ndarray) -> np.ndarray:
    """
    Mean-pool a processed sequence of shape (T, 132) into shape (132,).
    """
    return sequence.mean(axis=0).astype(np.float32)


def load_logistic_regression_model(model_path: Path = BASELINE_MODEL_PATH):
    """
    Load the saved logistic regression model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    return joblib.load(model_path)


def predict_landing_quality_from_video(
    video_path: Union[str, Path],
    model=None,
    target_length: int = SEQUENCE_LENGTH,
) -> Dict:
    """
    Run end-to-end inference on a raw video and return landing-quality prediction.
    """
    video_path = Path(video_path)

    if model is None:
        model = load_logistic_regression_model()

    frames, frame_indices = load_sampled_frames(
        video_path=video_path,
        num_samples=target_length,
        convert_bgr_to_rgb=True,
    )

    pose_estimator = initialize_pose_estimator()
    try:
        landmark_sequence, _ = extract_pose_sequence(frames, pose_estimator)
    finally:
        pose_estimator.close()

    missing_pose_frames = count_missing_pose_frames(landmark_sequence)

    processed_sequence = preprocess_landmark_sequence(
        landmark_sequence,
        target_length=target_length,
    )

    feature_vector = aggregate_sequence_mean(processed_sequence).reshape(1, -1)

    pred_index = int(model.predict(feature_vector)[0])

    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(feature_vector)[0]
        confidence = float(np.max(pred_proba))
        probabilities = {
            INDEX_TO_LABEL[i]: float(pred_proba[i]) for i in range(len(pred_proba))
        }
    else:
        confidence = None
        probabilities = None

    pred_label = INDEX_TO_LABEL[pred_index]

    return {
        "video_path": str(video_path),
        "predicted_label": pred_label,
        "predicted_index": pred_index,
        "confidence": confidence,
        "probabilities": probabilities,
        "num_sampled_frames": len(frame_indices),
        "missing_pose_frames": int(missing_pose_frames),
        "processed_sequence_shape": tuple(processed_sequence.shape),
    }