# JumpSafe Continuation: Pose-Based Jump-Landing Quality Classification

## Project Overview

JumpSafe Continuation is a pose-based machine learning project for assessing jump-landing quality from consumer video. This project extends earlier JumpSafe work by shifting from raw frame image classification to a more interpretable pipeline based on human pose landmarks and clip-level sequence modeling.

The current prototype processes a jump video through the following stages:

1. video loading and frame sampling  
2. pose estimation with MediaPipe Pose  
3. landmark cleaning and normalization  
4. fixed-length temporal sequence construction  
5. GRU-based sequence classification  
6. interpretable output through pose overlays and simple biomechanical summaries  

The project is intentionally scoped as a **binary classification task**:

- `bad_jump`
- `good_jump`

## Current Project Purpose

The main goal of this continuation project is to demonstrate an end-to-end, pose-based temporal modeling pipeline for movement quality assessment.

The project emphasizes three contributions:

- a **temporal deep learning pipeline** centered on a GRU sequence model
- an **interpretable pose-based representation** using landmark sequences instead of raw pixels
- a **working prototype interface** built with Streamlit for end-to-end video inference

This project is a **proof of concept** and should not be interpreted as a clinical or diagnostic injury-risk system.


## Current Pipeline

Raw Jump Video
→ Frame Sampling
→ MediaPipe Pose Extraction
→ Landmark Cleaning + Normalization
→ Fixed-Length Sequence Construction (30, 132)
→ Model Inference
   ├── **Primary model:** GRU temporal classifier
   └── **Baseline model:** Logistic regression baseline on mean-pooled features
→ Landing-Quality Prediction
→ Pose Overlay Visualization in Streamlit
→ Simple Biomechanical Summary

Each processed clip is represented as a fixed-length pose sequence of shape `(30, 132)`:
- `30` uniformly sampled frames per clip
- `33` pose landmarks per frame
- `4` values per landmark: `(x, y, z, visibility)`
- `33 × 4 = 132` features per frame

## Repository Structure
```text
JumpSafe_Continuation/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── metadata/
│   └── sample_videos/
├── docs/
├── models/
│   ├── gru_final.pt
│   ├── gru_final_config.json
│   └── logreg_model.joblib
├── notebooks/
│   ├── 01_environment_and_video_check.ipynb
│   ├── 02_pose_pipeline_debug.ipynb
│   ├── 03_build_landmark_sequences.ipynb
│   ├── 04_gru_model_evaluation.ipynb
│   ├── 05_static_baselines.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_gru_refinement.ipynb
│   └── setup.ipynb
├── results/
│   ├── figures/
│   ├── outputs/
│   ├── deliverable3_baseline/
│   └── deliverable3_tuning/
├── src/
│   ├── __init__.py
│   ├── biomechanics.py
│   ├── config.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── pose_extraction.py
│   ├── preprocessing.py
│   ├── sequence_builder.py
│   ├── temporal_model.py
│   ├── train_baselines.py
│   ├── train_temporal.py
│   ├── utils.py
│   ├── video_utils.py
│   └── visualization.py
└── ui/
    └── app.py
```
Note: Some legacy scaffold files may still remain in the repository for the timebeing, but the active workflow for the current prototype is based on the notebooks and source files listed above.

## Installation and Setup

### 1. Clone the repository
git clone https://github.com/armd08-cyber/jumpsafe-continuation.git
cd jumpsafe-continuation

### 2. Create and activate an environment

On macOS/Linux:
conda create -n jumpsafe312 python=3.12 -y
conda activate jumpsafe312

Or use your preferred Conda environment.

### 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4. Launch Jupyter
jupyter notebook

### 5. Launch Streamlit
streamlit run ui/app.py

### 6. Important compatibility note
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

### 7. 07_gru_refinement.ipynb
Runs:
- GRU refinement experiments
- final GRU training on the full processed dataset
- saving of the final deployed GRU model and config for inference/Streamlit use

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
- `07_gru_refinement.ipynb`

This evaluates the primary temporal model using stratified 5-fold cross-validation, then trains and saves a final GRU model for deployment.

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
- sampled pose-overlayed frames for interface display
- simple biomechanical summary values

### F. Run the Streamlit prototype
Launch:
- Before running the Streamlit app, make sure models/gru_final.pt and models/gru_final_config.json have been generated by 07_gru_refinement.ipynb
- streamlit run ui/app.py
The Streamlit app currently supports:
- video upload
- raw-video GRU inference
- pose-overlay frame visualization
- simple biomechanical summary display


## Current Results

### Evaluation approach

The main evaluation design uses **stratified 5-fold cross-validation at the clip level** to reduce leakage risk and provide more stable estimates than a single train/test split.

### Current modeling status

At this stage:

- the **GRU is the primary model** for the project
- logistic regression is retained as a **baseline**
- the end-to-end pipeline is functional from raw video to prediction
- the deployed GRU can run successfully inside the Streamlit interface
- model predictions remain somewhat unstable because of the small dataset

### Example comparison

The Deliverable 3 comparison table showed improvement over the earlier GRU baseline:

| Metric | GRU Deliverable 2 Baseline | GRU Deliverable 3 Tuned | Logistic Regression |
|---|---:|---:|---:|
| Accuracy | 0.500 ± 0.204 | 0.633 ± 0.139 | 0.633 ± 0.075 |
| Precision | 0.390 ± 0.261 | 0.590 ± 0.102 | 0.673 ± 0.192 |
| Recall | 0.600 ± 0.435 | 0.933 ± 0.149 | 0.667 ± 0.333 |
| F1 | 0.471 ± 0.325 | 0.719 ± 0.107 | 0.613 ± 0.157 |

### Interpretation of current performance

Current results should be interpreted cautiously:

- the prototype demonstrates that the full pose-based temporal inference pipeline works
- the GRU is technically integrated and deployable
- prediction quality is still limited by the small sample size and possible class instability
- outputs are best treated as **prototype estimates**, not definitive assessments

## Interface Status

A working Streamlit prototype is implemented in `ui/app.py`.

### Current interface capabilities

The app can:

- accept an uploaded jump video
- run the full backend inference pipeline
- display a GRU-based landing-quality prediction
- show confidence and class probabilities
- display sampled pose-overlay frames for interpretability
- display simple biomechanical summary values

### Interface purpose

The interface is intentionally lightweight and is meant to demonstrate:

- practical end-to-end deployment
- interpretable pose-based inference
- proof-of-concept user interaction with the GRU pipeline

## Interpretability Features

One of the main strengths of the project is its interpretable pose-based design.

### Current interpretability elements include:

- human pose landmark extraction instead of raw-pixel modeling
- explicit fixed-length landmark sequences
- pose-overlay frame visualization in the Streamlit app
- readable output fields such as confidence, class probabilities, and missing-pose-frame counts
- simple biomechanical summary values including:
  - landing frame index
  - left knee angle
  - right knee angle
  - mean knee angle
  - knee symmetry difference

These features help make the model pipeline more understandable than a purely black-box image classifier.

## Known Issues and Current Limitations

- The dataset is very small (`30` clips), so performance estimates remain unstable.
- The GRU is functional, but predictions on individual clips are still somewhat inconsistent.
- Pose extraction quality can vary depending on video quality, viewpoint, lighting, and body visibility.
- The current system may still show class bias or weak generalization on unseen clips.
- Labels are binary and simplified for project scope, which limits nuance.
- Formal inter-rater agreement validation has not yet been completed.
- The project is not a clinical or diagnostic tool and should not be used for medical decision-making.

## Planned Next Steps

Planned improvements include:

- further GRU refinement and error analysis
- deeper fold-level performance review
- confusion-matrix interpretation
- improved result presentation in Streamlit
- optional label-quality strengthening through a second-rater subset and agreement analysis
- continued emphasis on careful proof-of-concept framing

## Author

Arjun Rammohandas

## Contact

a.rammohandas@ufl.edu