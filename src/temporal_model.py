import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Simple GRU-based classifier for jump landmark sequences.
    """

    def __init__(
        self,
        input_dim: int = 132,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_classes)
        """
        output, hidden = self.gru(x)

        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)
        return logits