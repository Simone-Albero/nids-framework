import numpy as np
from numpy.typing import NDArray
import torch

from .base import Metric


class MulticlassClassificationMetric(Metric):
    __slots__ = [
        "_class_stats",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    def __init__(self, n_classes: int) -> None:
        super().__init__(n_classes)
        self._class_stats: NDArray = np.zeros((n_classes, 3))
        self.weighted_precision: float = 0.0
        self.weighted_recall: float = 0.0
        self.weighted_f1: float = 0.0

    def step(self, y: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = torch.argmax(y, dim=1)
        self.update(y_pred, y_true)

    def compute_metrics(self) -> None:
        tp = np.diag(self._confusion_matrix)
        fp = np.sum(self._confusion_matrix, axis=0) - tp
        fn = np.sum(self._confusion_matrix, axis=1) - tp

        precision = np.divide(
            tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
        )
        recall = np.divide(
            tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
        )
        f1_score = np.divide(
            2 * precision * recall,
            (precision + recall),
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        self._class_stats[:, 0] = precision
        self._class_stats[:, 1] = recall
        self._class_stats[:, 2] = f1_score

        support = np.sum(self._confusion_matrix, axis=1)
        total_support = np.sum(support)

        self.weighted_precision = np.sum(precision * support) / total_support
        self.weighted_recall = np.sum(recall * support) / total_support
        self.weighted_f1 = np.sum(f1_score * support) / total_support

    def __str__(self) -> str:
        output = [
            f"Confusion Matrix:\n{self._confusion_matrix}\n",
            "Class-wise metrics:\n",
        ]

        for i in range(self._n_classes):
            precision, recall, f1 = self._class_stats[i]
            support = np.sum(self._confusion_matrix[i, :])
            output.append(
                f"Class {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Support: {support}\n"
            )

        output.append(f"\nWeighted Precision: {self.weighted_precision:.4f}")
        output.append(f"\nWeighted Recall: {self.weighted_recall:.4f}")
        output.append(f"\nWeighted F1 Score: {self.weighted_f1:.4f}\n")

        total_support = np.sum(np.sum(self._confusion_matrix, axis=1))
        output.append(f"\nTotal Support: {total_support}\n")

        return "".join(output)

    def save(
        self, file_path: str = "logs/multiclass_metrics.csv", reset_logs: bool = False
    ) -> None:
        super().save(file_path, reset_logs)

    def _write_header(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            headers = ["Class", "Precision", "Recall", "F1", "Support"]
            f.write(",".join(headers) + "\n")

    def _write_metrics(self, file_handle) -> None:
        for i in range(self._n_classes):
            precision, recall, f1 = self._class_stats[i]
            support = np.sum(self._confusion_matrix[i, :])
            file_handle.write(f"{i},{precision:.4f},{recall:.4f},{f1:.4f},{support}\n")

        total_support = np.sum(np.sum(self._confusion_matrix, axis=1))
        file_handle.write(
            f"Weighted,{self.weighted_precision:.3f},{self.weighted_recall:.3f},{self.weighted_f1:.3f},{total_support}\n"
        )
