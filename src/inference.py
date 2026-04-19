from pathlib import Path
from typing import Dict, Union
import json

import numpy as np
import torch
import torch.nn.functional as F

from src.biomechanics import summarize_biomechanics
from src.config import INDEX_TO_LABEL, SEQUENCE_LENGTH
from src.video_utils import load_sampled_frames
from src.pose_extraction import (
    initialize_pose_estimator,
    extract_pose_sequence,
    count_missing_pose_frames,
)
from src.preprocessing import preprocess_landmark_sequence
from src.temporal_model import GRUClassifier
from src.visualization import generate_pose_overlay_frames

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRU_MODEL_PATH = PROJECT_ROOT / "models" / "gru_final.pt"
GRU_CONFIG_PATH = PROJECT_ROOT / "models" / "gru_final_config.json"


def load_gru_model(
    model_path: Path = GRU_MODEL_PATH,
    config_path: Path = GRU_CONFIG_PATH,
):
    """
    Load saved GRU model and config.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Saved GRU model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Saved GRU config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model = GRUClassifier(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def predict_landing_quality_from_video(
    video_path: Union[str, Path],
    model=None,
    target_length: int = SEQUENCE_LENGTH,
) -> Dict:
    """
    Run end-to-end GRU inference on a raw video and return prediction details.
    """
    video_path = Path(video_path)

    if model is None:
        model, model_config = load_gru_model()
    else:
        model_config = None

    frames_bgr, frame_indices = load_sampled_frames(
        video_path=video_path,
        num_samples=target_length,
        convert_bgr_to_rgb=False,
    )

    frames_rgb = [frame[:, :, ::-1] for frame in frames_bgr]

    pose_estimator = initialize_pose_estimator()
    try:
        landmark_sequence, pose_results = extract_pose_sequence(frames_rgb, pose_estimator)
    finally:
        pose_estimator.close()

    missing_pose_frames = count_missing_pose_frames(landmark_sequence)

    processed_sequence = preprocess_landmark_sequence(
        landmark_sequence,
        target_length=target_length,
    ).astype(np.float32)

    biomechanical_summary = summarize_biomechanics(processed_sequence)

    input_tensor = torch.tensor(processed_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_index = int(np.argmax(probs))
    pred_label = INDEX_TO_LABEL[pred_index]
    confidence = float(np.max(probs))

    probabilities = {
        INDEX_TO_LABEL[i]: float(probs[i]) for i in range(len(probs))
    }

    pose_overlay_frames = generate_pose_overlay_frames(frames_bgr, max_frames=4)

    return {
        "video_path": str(video_path),
        "predicted_label": pred_label,
        "predicted_index": pred_index,
        "confidence": confidence,
        "probabilities": probabilities,
        "num_sampled_frames": len(frame_indices),
        "missing_pose_frames": int(missing_pose_frames),
        "processed_sequence_shape": tuple(processed_sequence.shape),
        "pose_overlay_frames": pose_overlay_frames,
        "biomechanical_summary": biomechanical_summary,
        "model_type": "GRU",
        "model_path": str(GRU_MODEL_PATH),
        "model_config": model_config,
    }