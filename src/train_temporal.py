from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, RANDOM_SEED
from src.dataset import JumpSequenceDataset, load_processed_metadata
from src.temporal_model import GRUClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CV_RESULTS_PATH = PROCESSED_DIR / "cv_results.csv"
CV_PREDICTIONS_PATH = PROCESSED_DIR / "cv_predictions.csv"
CV_SUMMARY_PATH = PROCESSED_DIR / "cv_summary.csv"
CV_CONFUSION_MATRIX_PATH = PROCESSED_DIR / "cv_confusion_matrix.npy"


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = JumpSequenceDataset(train_df)
    val_dataset = JumpSequenceDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels, _ in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device = DEVICE,
) -> Dict:
    model.eval()
    running_loss = 0.0
    total = 0

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_clip_ids: List[str] = []

    with torch.no_grad():
        for sequences, labels, clip_ids in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)

            running_loss += loss.item() * sequences.size(0)
            total += labels.size(0)

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_clip_ids.extend(list(clip_ids))

    avg_loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": all_labels,
        "preds": all_preds,
        "clip_ids": all_clip_ids,
    }


def run_single_split_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
) -> Dict:
    train_loader, val_loader = build_dataloaders(train_df, val_df, batch_size=batch_size)

    model = GRUClassifier(
        input_dim=132,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_classes=2,
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=DEVICE)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion, device=DEVICE)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

    final_val_metrics = evaluate_one_epoch(model, val_loader, criterion, device=DEVICE)

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "final_val_metrics": final_val_metrics,
    }


def run_stratified_kfold_cv(
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    random_state: int = RANDOM_SEED,
    save_results: bool = True,
) -> Dict[str, pd.DataFrame]:
    df = metadata_df.reset_index(drop=True).copy()
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_records = []
    prediction_records = []
    aggregate_confusion_matrix = np.zeros((2, 2), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\nStarting fold {fold_idx}/{n_splits}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        result = run_single_split_experiment(
            train_df=train_df,
            val_df=val_df,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )

        metrics = result["final_val_metrics"]
        aggregate_confusion_matrix += metrics["confusion_matrix"]

        fold_record = {
            "fold": fold_idx,
            "val_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        fold_records.append(fold_record)

        for clip_id, true_label, pred_label in zip(
            metrics["clip_ids"], metrics["labels"], metrics["preds"]
        ):
            prediction_records.append(
                {
                    "fold": fold_idx,
                    "clip_id": clip_id,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": int(true_label == pred_label),
                }
            )

    results_df = pd.DataFrame(fold_records)
    predictions_df = pd.DataFrame(prediction_records)

    metric_cols = ["val_loss", "accuracy", "precision", "recall", "f1"]
    summary_df = pd.DataFrame(
        {
            "metric": metric_cols,
            "mean": [results_df[col].mean() for col in metric_cols],
            "std": [results_df[col].std() for col in metric_cols],
        }
    )

    if save_results:
        CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(CV_RESULTS_PATH, index=False)
        predictions_df.to_csv(CV_PREDICTIONS_PATH, index=False)
        summary_df.to_csv(CV_SUMMARY_PATH, index=False)
        np.save(CV_CONFUSION_MATRIX_PATH, aggregate_confusion_matrix)

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
    results = run_stratified_kfold_cv(metadata_df=metadata, n_splits=5, num_epochs=10)

    print("\nFold results:")
    print(results["fold_results"])

    print("\nSummary:")
    print(results["summary"])

    print("\nAggregate confusion matrix:")
    print(results["confusion_matrix"])