# JumpSafe Continuation: Pose-Based Jump-Landing Quality Classification

## Project Overview
JumpSafe Continuation is a pose-based machine learning project for assessing jump-landing quality from consumer video. This project extends earlier JumpSafe work by shifting from raw frame image classification to a more interpretable pipeline based on human pose landmarks and clip-level sequence modeling.

The current prototype takes short labeled jump videos, samples frames, extracts pose landmarks with MediaPipe Pose, cleans and normalizes landmark sequences, and trains clip-level classifiers to predict a landing-quality label (`good_jump` or `bad_jump`). The project includes both a primary temporal model (GRU) and a simpler static baseline (logistic regression on mean-pooled pose features).

## Current Project Purpose
The goal of the current stage is to build and verify an end-to-end prototype that can:

- process raw jump videos into standardized pose-based representations
- train and evaluate a temporal sequence model for landing-quality classification
- compare the temporal model against a simpler static baseline
- generate landing-quality predictions from raw video input
- support a future lightweight interface layer for user-facing inference

This project is a proof of concept and should not be interpreted as a clinical or diagnostic injury-risk system.

## Current Pipeline

Raw Jump Video
→ Frame Sampling
→ MediaPipe Pose Extraction
→ Landmark Cleaning + Normalization
→ Fixed-Length Sequence Construction (30, 132)
→ Model Inference
   ├── GRU temporal classifier
   └── Logistic regression baseline on mean-pooled features
→ Landing-Quality Prediction

Each processed clip is represented as a fixed-length pose sequence of shape `(30, 132)`:
- `30` uniformly sampled frames per clip
- `33` pose landmarks per frame
- `4` values per landmark: `(x, y, z, visibility)`
- `33 × 4 = 132` features per frame

## Repository Structure

JumpSafe_Continuation/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample_videos/
│
├── notebooks/
│   ├── 01_environment_and_video_check.ipynb
│   ├── 02_pose_pipeline_debug.ipynb
│   ├── 03_build_landmark_sequences.ipynb
│   ├── 04_gru_model_evaluation.ipynb
│   ├── 05_static_baselines.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── video_utils.py
│   ├── pose_extraction.py
│   ├── preprocessing.py
│   ├── sequence_builder.py
│   ├── dataset.py
│   ├── temporal_model.py
│   ├── train_temporal.py
│   ├── train_baselines.py
│   └── inference.py
│
├── ui/
│   └── app.py
│
├── results/
│   └── figures/
│
├── docs/
├── README.md
├── requirements.txt
└── .gitignore

Note: Some legacy scaffold files may still remain in the repository for the timebeing, but the active workflow for the current prototype is based on the notebooks and source files listed above.

## Installation and Setup

### 1. Clone the repository
git clone https://github.com/armd08-cyber/jumpsafe-continuation.git
cd jumpsafe-continuation

### 2. Create and activate an environment

On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

Or use your preferred Conda environment.

### 3. Install dependencies
pip install -r requirements.txt

### 4. Launch Jupyter
jupyter notebook

### 5. Important compatibility note
This project currently uses `mediapipe==0.10.21` for compatibility with the implemented `mp.solutions.pose` pipeline.

## Recommended Notebook Order

Run the notebooks in this order after a fresh kernel restart.

### 1. 01_environment_and_video_check.ipynb
Verifies:
- imports and environment setup
- metadata loading
- raw video path validation
- frame sampling from a sample clip

### 2. 02_pose_pipeline_debug.ipynb
Verifies:
- MediaPipe Pose extraction
- landmark sequence generation
- preprocessing into fixed-length `(30, 132)` clip representations

### 3. 03_build_landmark_sequences.ipynb
Builds:
- processed `.npy` landmark sequences for all clips
- processed sequence metadata

### 4. 04_gru_model_evaluation.ipynb
Runs:
- GRU dataset loading
- GRU forward-pass sanity checks
- single-split sanity training
- stratified 5-fold cross-validation for the GRU model

### 5. 05_static_baselines.ipynb
Runs:
- logistic regression baseline on mean-pooled clip-level pose features
- stratified 5-fold cross-validation
- final logistic regression model saving

### 6. 06_model_comparison.ipynb
Builds:
- side-by-side comparison table between GRU and logistic regression

## Dataset Information

The current dataset consists of:
- `30` labeled raw jump clips
- balanced binary labels:
  - `15` `good_jump`
  - `15` `bad_jump`

The raw videos are stored locally under `data/raw/`, and processed clip representations are saved under `data/processed/`.

Due to the dataset's small size, evaluation is performed at the clip level using stratified 5-fold cross-validation rather than a single random holdout split.

## How to Reproduce the Current Results

### A. Build processed pose sequences
Run:
- `03_build_landmark_sequences.ipynb`

This generates:
- processed landmark sequences in `data/processed/landmark_sequences/`
- processed metadata CSV in `data/processed/`

### B. Run the GRU temporal model
Run:
- `04_gru_model_evaluation.ipynb`

This evaluates the primary temporal model using stratified 5-fold cross-validation and saves GRU evaluation outputs.

### C. Run the logistic regression baseline
Run:
- `05_static_baselines.ipynb`

This evaluates the static baseline and saves:
- fold-level metrics
- predictions
- confusion matrix
- final saved logistic regression model (`logreg_model.joblib`)

### D. Compare models
Run:
- `06_model_comparison.ipynb`

This generates a side-by-side comparison table between the GRU and logistic regression baseline.

### E. Test raw-video inference
A working backend inference path is implemented in:
- `src/inference.py`

It can process a raw jump video and return:
- predicted label
- confidence
- class probabilities
- missing-pose-frame count
- processed sequence shape

## Current Results

### Preliminary cross-validation summary
Under the current implementation:
- the logistic regression baseline outperformed the GRU in preliminary stratified 5-fold cross-validation
- the GRU pipeline is functional, but its current performance is less stable under the small-data setting

### Model comparison
Current mean ± standard deviation results:

| Metric | GRU | Logistic Regression |
|---|---|---|
| Accuracy | 0.500 ± 0.204 | 0.633 ± 0.075 |
| Precision | 0.390 ± 0.261 | 0.673 ± 0.192 |
| Recall | 0.600 ± 0.435 | 0.667 ± 0.333 |
| F1 Score | 0.471 ± 0.325 | 0.613 ± 0.157 |

### Sample inference
A sample raw-video inference run successfully produced:
- predicted label: `bad_jump`
- confidence: `0.857`
- processed sequence shape: `(30, 132)`

These results should be interpreted as preliminary proof-of-concept findings rather than final performance claims.

## Interface Status

A full user-facing interface is planned but not yet finalized in the current deliverable.

Current status:
- the backend inference pipeline is functional
- a future lightweight interface (likely Streamlit) is intended to:
  - accept a jump video
  - run the pose-based inference pipeline
  - display a landing-quality prediction and supporting outputs

For this stage, the model-serving backend is complete enough to support future UI integration, and `ui/app.py` is reserved for that next phase of development.

## Known Issues and Current Limitations

- The dataset is very small (`30` clips), so model performance estimates remain unstable.
- The GRU currently underperforms the logistic regression baseline under the present settings.
- Pose extraction quality can vary depending on video quality, viewpoint, and visibility.
- The interface layer is still in progress.
- The project is not a clinical or diagnostic tool and should not be used for medical decision-making.

## Planned Next Steps

Before the next deliverable, planned improvements include:
- hyperparameter tuning for the GRU
- possible comparison with an LSTM variant
- deeper error analysis and confusion-matrix review
- additional label-quality checks and possible second-rater validation
- refinement of the inference interface
- improved interpretability outputs such as pose overlays or simple biomechanical summaries

## Author

Arjun Rammohandas

## Contact

a.rammohandas@ufl.edu