import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


def format_label(label: str) -> str:
    """
    Convert labels like 'bad_jump' to 'Bad Jump'.
    """
    return label.replace("_", " ").title()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import predict_landing_quality_from_video

st.set_page_config(page_title="JumpSafe Continuation", layout="wide")

st.title("JumpSafe Continuation")
st.subheader("Pose-Based Jump Landing Quality Assessment")

st.write(
    "This prototype accepts a short jump-landing video, extracts pose landmarks, "
    "constructs a fixed-length temporal sequence, and predicts landing quality "
    "using a GRU-based sequence model."
)

st.info(
    "Proof-of-concept only. This system is intended for coursework and prototype "
    "demonstration purposes, not for clinical or diagnostic use."
)

model_path = PROJECT_ROOT / "models" / "gru_final.pt"
config_path = PROJECT_ROOT / "models" / "gru_final_config.json"

if model_path.exists() and config_path.exists():
    st.success(f"GRU model found: {model_path.name}")
else:
    st.error("GRU model files not found. Please train and save the final GRU model first.")

uploaded_file = st.file_uploader(
    "Upload a jump video",
    type=["mov", "mp4", "avi", "m4v"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Run Landing Quality Prediction"):
        with st.spinner("Running pose extraction and GRU inference..."):
            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = Path(tmp_file.name)

            try:
                result = predict_landing_quality_from_video(temp_video_path)

                pred_label = result["predicted_label"]
                confidence = result["confidence"]
                model_type = result["model_type"]

                st.subheader("Prediction Result")

                formatted_prediction = format_label(pred_label)
                if pred_label == "good_jump":
                    st.success(f"Predicted label: {formatted_prediction}")
                else:
                    st.warning(f"Predicted label: {formatted_prediction}")

                st.write(f"**Confidence:** {confidence:.3f}")
                st.write(f"**Model type:** {model_type}")
                st.write(f"**Sampled frames:** {result['num_sampled_frames']}")
                st.write(f"**Missing pose frames:** {result['missing_pose_frames']}")
                st.write(f"**Processed sequence shape:** {result['processed_sequence_shape']}")

                probs = result.get("probabilities", {})
                if probs:
                    st.subheader("Class Probabilities")

                    prob_df = pd.DataFrame({
                        "Class": [format_label(label) for label in probs.keys()],
                        "Probability": [round(prob, 3) for prob in probs.values()],
                    })
                    st.table(prob_df)

                summary = result.get("biomechanical_summary", {})
                if summary:
                    st.subheader("Biomechanical Summary")
                    st.write(f"**Landing frame index:** {summary['landing_frame_index']}")
                    st.write(f"**Left knee angle:** {summary['left_knee_angle']:.1f}°")
                    st.write(f"**Right knee angle:** {summary['right_knee_angle']:.1f}°")
                    st.write(f"**Mean knee angle:** {summary['mean_knee_angle']:.1f}°")
                    st.write(f"**Knee symmetry difference:** {summary['knee_symmetry_diff']:.1f}°")

                    st.caption(
                        "These are pose-derived movement summary values from the sampled "
                        "landmark sequence. They are intended for interpretability and "
                        "prototype demonstration only."
                    )

                overlay_frames = result.get("pose_overlay_frames", [])
                if overlay_frames:
                    st.subheader("Pose Overlay Samples")
                    st.write(
                        "These sampled frames show pose landmarks detected from the uploaded video."
                    )

                    cols = st.columns(len(overlay_frames))
                    for i, frame in enumerate(overlay_frames):
                        with cols[i]:
                            st.image(
                                frame,
                                caption=f"Overlay Frame {i+1}",
                                use_container_width=True,
                            )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

            finally:
                if temp_video_path.exists():
                    temp_video_path.unlink()