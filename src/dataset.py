from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import LABEL_TO_INDEX, PROCESSED_DIR


PROCESSED_METADATA_PATH = PROCESSED_DIR / "processed_sequence_metadata.csv"


class JumpSequenceDataset(Dataset):
    """
    PyTorch dataset for processed jump landmark sequences.
    """

    def __init__(self, metadata_df: pd.DataFrame):
        self.metadata_df = metadata_df.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.metadata_df.iloc[idx]

        sequence = np.load(row["sequence_path"]).astype(np.float32)
        label = LABEL_TO_INDEX[row["label"]]
        clip_id = row["clip_id"]

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, label_tensor, clip_id


def load_processed_metadata(metadata_path: Path = PROCESSED_METADATA_PATH) -> pd.DataFrame:
    """
    Load processed sequence metadata CSV.
    """
    df = pd.read_csv(metadata_path)

    required_columns = {"clip_id", "label", "sequence_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df