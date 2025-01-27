import os

import numpy as np
from numpy.typing import NDArray
import torch
from sklearn.metrics import confusion_matrix


class Metric:
    __slots__ = [
        "_confusion_matrix",
        "_n_classes",
    ]

    def __init__(self, n_classes: int) -> None:
        self._confusion_matrix: NDArray = np.zeros((n_classes, n_classes), dtype=int)
        self._n_classes: int = n_classes

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        batch_confusion_matrix = confusion_matrix(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            labels=np.arange(self._n_classes),
        )
        self._confusion_matrix += batch_confusion_matrix

    def step(self, y: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def compute_metrics(self) -> None:
        pass

    def __str__(self) -> str:
        pass

    def save(
        self,
        file_path: str = "logs/metrics.csv",
        reset_logs: bool = False,
    ) -> None:
        if reset_logs or not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._write_header(file_path)

        with open(file_path, "a") as f:
            self._write_metrics(f)

    def _write_header(self, file_path: str) -> None:
        pass

    def _write_metrics(self, file_handle) -> None:
        pass


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
        TP = self._confusion_matrix[1, 1]
        FP = self._confusion_matrix[0, 1]
        FN = self._confusion_matrix[1, 0]
        TN = self._confusion_matrix[0, 0]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        self.precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        self.recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
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

    def save(
        self,
        file_path: str = "logs/binary_metrics.csv",
        reset_logs: bool = False,
    ) -> None:
        super().save(file_path, reset_logs)

    def _write_header(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            headers = ["Precision", "Recall", "F1", "Balanced Accuracy", "TN", "FN", "FP", "TP"]
            f.write(",".join(headers) + "\n")

    def _write_metrics(self, file_handle) -> None:
        file_handle.write(
        f"{self.precision:.3f},{self.recall:.3f},{self.f1:.3f},{self.balanced_accuracy:.3f},"
        f"{self._confusion_matrix[0, 0]},{self._confusion_matrix[0, 1]},"
        f"{self._confusion_matrix[1, 0]},{self._confusion_matrix[1, 1]}\n"
    )


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
        TP = np.diag(self._confusion_matrix)
        FP = np.sum(self._confusion_matrix, axis=0) - TP
        FN = np.sum(self._confusion_matrix, axis=1) - TP

        precision = np.divide(
            TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0
        )
        recall = np.divide(
            TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0
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
        self,
        file_path: str = "logs/multiclass_metrics.csv",
        reset_logs: bool = False,
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
