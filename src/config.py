from pathlib import Path

# Project root: JumpSafe_Continuation/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
UI_DIR = PROJECT_ROOT / "ui"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset metadata
CLIP_METADATA_PATH = PROCESSED_DIR / "clip_metadata.csv"

# Sequence / landmark settings
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
LANDMARK_DIMS = 4
FEATURE_DIM = NUM_LANDMARKS * LANDMARK_DIMS  # 33 * 4 = 132

# Class information
CLASS_NAMES = ["bad_jump", "good_jump"]
LABEL_TO_INDEX = {
    "bad_jump": 0,
    "good_jump": 1,
}
INDEX_TO_LABEL = {
    0: "bad_jump",
    1: "good_jump",
}

# Reproducibility
RANDOM_SEED = 42