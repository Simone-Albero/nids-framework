import os
import numpy as np
from numpy.typing import NDArray
import torch

class Metric:
    __slots__ = ["_confusion_matrix", "_n_classes"]

    def __init__(self, n_classes: int) -> None:
        self._confusion_matrix: NDArray = np.zeros((n_classes, n_classes), dtype=int)
        self._n_classes: int = n_classes

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred_np = y_pred.cpu().numpy().ravel().astype(int)
        y_true_np = y_true.cpu().numpy().ravel().astype(int)

        np.add.at(self._confusion_matrix, (y_true_np, y_pred_np), 1)

    def step(self, y: torch.Tensor, y_true: torch.Tensor) -> None:
        """Handle logic for step-by-step metric computation."""
        pass

    def compute_metrics(self) -> None:
        """Calculate metrics after updates."""
        pass

    def __str__(self) -> str:
        """String representation of metrics."""
        pass

    def save(self, file_path: str = "logs/metrics.csv", reset_logs: bool = False) -> None:
        if reset_logs or not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._write_header(file_path)

        with open(file_path, "a") as f:
            self._write_metrics(f)

    def _write_header(self, file_path: str) -> None:
        """Write CSV header."""
        pass

    def _write_metrics(self, file_handle) -> None:
        """Write computed metrics to CSV."""
        pass
