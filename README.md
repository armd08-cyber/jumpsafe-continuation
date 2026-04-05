# JumpSafe Continuation: Pose-Based Jump-Landing Quality Classification

## Project Overview
This project is a continuation of JumpSafe, an embedded AI system for jump classification using consumer video. This continuation shifts from raw frame image classification to a pose-based machine learning pipeline for assessing jump landing quality using interpretable biomechanical features. The goal is to build a proof-of-concept system that can accept a short jump video, extracts thepose landmarks, engineers landing related features, and predicts a simple landing quality label.


## Project Goals
The goals of this project are to:
- organize and validate the JumpSafe continuation dataset and environment
- extract pose landmarks and temporal landmark sequences from jump videos
- engineer interpretable biomechanical features related to landing quality
- train a primary temporal deep learning model, such as an LSTM or GRU, for clip-level landing-quality classification
- compare the temporal model against simpler static baselines
- develop a simple interactive interface for demonstrating inference results

## Repository Structure

```text
JumpSafe_Continuation/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample_videos/
├── notebooks/
│   └── setup.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── pose_extraction.py
│   ├── sequence_builder.py
│   ├── feature_engineering.py
│   ├── train_lstm.py
│   ├── train_baselines.py
│   ├── evaluate.py
│   └── utils.py
├── ui/
│   └── app.py
├── results/
│   ├── figures/
│   └── outputs/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore

```

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/armd08-cyber/jumpsafe-continuation.git
cd jumpsafe-continuation
```

### 2. Create and activate a virtual environment

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Notebook

From the project root, launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:
`notebooks/setup.ipynb`

The setup notebook is intended to:
- verify that the environment is functioning as intended
- confirm dataset loading/access
- perform early data exploration
- generate visible output like summary tables, plots, and a sample video frame

## Dataset Information

This project uses the existing JumpSafe dataset collected in a controlled indoor setting. The dataset currently includes 30 raw jump videos stored locally in `data/raw/` and 2 labeled sample videos stored in `data/sample_videos/` for setup and testing purposes. The project uses short jump video clips as the primary data source and will later derive pose landmarks and engineered biomechanical features from those videos.

Because the raw dataset consists of video files, the repository structure separates data into:

- `data/raw/` for original source videos
- `data/interim/` for intermediate outputs such as extracted landmarks
- `data/processed/` for model-ready feature tables
- `data/sample_videos/` for small demo clips used in setup and interface testing

Large raw video files may be kept locally rather than fully tracked in GitHub, depending on repository size constraints. 

Because the dataset contains only 30 labeled clips, model evaluation is planned at the clip level using leave-one-out cross-validation (LOOCV) or stratified k-fold cross-validation rather than a single holdout split. A subset of clips will also be independently reviewed by a second rater so that inter-rater agreement can be estimated using Cohen’s kappa.


## Planned Pipeline
The planned end-to-end pipeline is:

```text
Jump Video → Frame Sampling → Pose Estimation → Landmark Processing
→ Landmark Sequence Construction → Temporal Model (LSTM / GRU)
→ Landing-Quality Prediction → UI Output
```

Static baseline models such as logistic regression, random forest, and a small multilayer perceptron will also be tested on aggregated pose-based features for comparison. The temporal model is the primary approach because jump-landing quality is inherently dynamic and depends on coordination across time rather than posture at a single instant.


## User Interface
Although the original JumpSafe system was implemented as an edge-oriented pipeline using an iPhone client, Flask server, and ESP32 device, the continuation project will use a lightweight Streamlit interface during the current development phase. This choice is intentional because the main focus of the continuation is to build and validate the new pose-based inference pipeline, not to rebuild a full mobile deployment stack immediately. Streamlit provides a practical way to connect video upload, preprocessing, pose extraction, feature generation, and model prediction within the same Python environment used for development.

The interface will allow a user to:
- upload a short jump video
- run the pose-based inference pipeline
- view a landing-quality prediction
- inspect selected interpretable outputs such as pose overlays or feature summaries

In this way, the Streamlit interface serves as a prototype demonstration layer rather than the final deployment target. The broader project still builds on the edge-AI and low cost deployment goals established by the original JumpSafe system, and a future version could reconnect the improved pose based backend to a mobile or embedded client if needed.

## Author

Arjun Rammohandas

## Contact

a.rammohandas@ufl.edu
