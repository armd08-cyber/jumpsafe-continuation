from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, RANDOM_SEED, LABEL_TO_INDEX
from src.dataset import load_processed_metadata


BASELINE_RESULTS_PATH = PROCESSED_DIR / "logreg_cv_results.csv"
BASELINE_PREDICTIONS_PATH = PROCESSED_DIR / "logreg_cv_predictions.csv"
BASELINE_SUMMARY_PATH = PROCESSED_DIR / "logreg_cv_summary.csv"
BASELINE_CONFUSION_MATRIX_PATH = PROCESSED_DIR / "logreg_cv_confusion_matrix.npy"
BASELINE_MODEL_PATH = PROCESSED_DIR / "logreg_model.joblib"


def aggregate_sequence_mean(sequence: np.ndarray) -> np.ndarray:
    """
    Mean-pool a sequence of shape (T, 132) into a vector of shape (132,).
    """
    return sequence.mean(axis=0).astype(np.float32)


def build_static_feature_matrix(metadata_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X, y, clip_ids from processed sequence metadata.
    """
    features = []
    labels = []
    clip_ids = []

    for _, row in metadata_df.iterrows():
        sequence = np.load(row["sequence_path"]).astype(np.float32)
        feature_vector = aggregate_sequence_mean(sequence)

        features.append(feature_vector)
        labels.append(LABEL_TO_INDEX[row["label"]])
        clip_ids.append(row["clip_id"])

    X = np.vstack(features)
    y = np.array(labels, dtype=np.int64)
    return X, y, clip_ids

def train_final_logistic_regression_model(metadata_df: pd.DataFrame, save_model: bool = True):
    """
    Train logistic regression on the full processed dataset and optionally save it.
    """
    X, y, clip_ids = build_static_feature_matrix(metadata_df)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )

    model.fit(X, y)

    if save_model:
        BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, BASELINE_MODEL_PATH)

    return model


def run_logistic_regression_cv(
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
    save_results: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run stratified k-fold CV for logistic regression on mean-pooled sequence features.
    """
    X, y, clip_ids = build_static_feature_matrix(metadata_df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_records = []
    prediction_records = []
    aggregate_confusion_matrix = np.zeros((2, 2), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

        aggregate_confusion_matrix += cm

        fold_records.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

        for i, idx in enumerate(val_idx):
            prediction_records.append(
                {
                    "fold": fold_idx,
                    "clip_id": clip_ids[idx],
                    "true_label": int(y[idx]),
                    "pred_label": int(y_pred[i]),
                    "correct": int(y[idx] == y_pred[i]),
                }
            )

    results_df = pd.DataFrame(fold_records)
    predictions_df = pd.DataFrame(prediction_records)

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    summary_df = pd.DataFrame(
        {
            "metric": metric_cols,
            "mean": [results_df[col].mean() for col in metric_cols],
            "std": [results_df[col].std() for col in metric_cols],
        }
    )

    if save_results:
        BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(BASELINE_RESULTS_PATH, index=False)
        predictions_df.to_csv(BASELINE_PREDICTIONS_PATH, index=False)
        summary_df.to_csv(BASELINE_SUMMARY_PATH, index=False)
        np.save(BASELINE_CONFUSION_MATRIX_PATH, aggregate_confusion_matrix)

    return {
        "fold_results": results_df,
        "predictions": predictions_df,
        "summary": summary_df,
        "confusion_matrix": pd.DataFrame(
            aggregate_confusion_matrix,
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        ),
    }


if __name__ == "__main__":
    metadata = load_processed_metadata()

    outputs = run_logistic_regression_cv(metadata_df=metadata, n_splits=5, save_results=True)

    print("\nFold results:")
    print(outputs["fold_results"])

    print("\nSummary:")
    print(outputs["summary"])

    print("\nConfusion matrix:")
    print(outputs["confusion_matrix"])

    final_model = train_final_logistic_regression_model(metadata, save_model=True)
    print(f"\nSaved final logistic regression model to: {BASELINE_MODEL_PATH}")