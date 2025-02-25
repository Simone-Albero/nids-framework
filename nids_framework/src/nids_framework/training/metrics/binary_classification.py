import torch

from .base import Metric

class BinaryClassificationMetric(Metric):
    __slots__ = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "_threshold",
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(n_classes=2)
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f1: float = 0.0
        self.balanced_accuracy: float = 0.0
        self._threshold: float = threshold

    def step(self, y: torch.Tensor, y_true: torch.Tensor) -> None:
        y_pred = y >= self._threshold
        self.update(y_pred, y_true)

    def compute_metrics(self) -> None:
        tp = self._confusion_matrix[1, 1]
        fp = self._confusion_matrix[0, 1]
        fn = self._confusion_matrix[1, 0]
        tn = self._confusion_matrix[0, 0]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1 = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        self.balanced_accuracy = (sensitivity + specificity) / 2

    def __str__(self) -> str:
        return (
            f"Confusion Matrix:\n{self._confusion_matrix}\n"
            f"Binary classification metrics:\n"
            f"Class 1 - Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1 Score: {self.f1:.4f}, Balanced Accuracy: {self.balanced_accuracy:.4f}\n"
        )

    def save(self, file_path: str = "logs/binary_metrics.csv", reset_logs: bool = False) -> None:
        super().save(file_path, reset_logs)

    def _write_header(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            headers = [
                "Precision",
                "Recall",
                "F1",
                "Balanced Accuracy",
                "TN",
                "FN",
                "FP",
                "TP",
            ]
            f.write(",".join(headers) + "\n")

    def _write_metrics(self, file_handle) -> None:
        file_handle.write(
            f"{self.precision:.3f},{self.recall:.3f},{self.f1:.3f},{self.balanced_accuracy:.3f},"
            f"{self._confusion_matrix[0, 0]},{self._confusion_matrix[0, 1]},"
            f"{self._confusion_matrix[1, 0]},{self._confusion_matrix[1, 1]}\n"
        )