from typing import Optional
import os

import torch


class Metric():

    __slots__ = [
        "_precision",
        "_recall",
        "_F1",
        "_TP",
        "_TN",
        "_FP",
        "_FN",
    ]

    def __init__(self) -> None:
        self._precision: float = 0
        self._recall: float = 0
        self._F1: float = 0
        self._TP: float = 0
        self._TN: float = 0
        self._FP: float = 0
        self._FN: float = 0

    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pass

    def apply(self) -> None:
        pass

    def __str__(self) -> str:
        return (
            f"Statistics:\n"
            f"Precision: {self._precision:.4f}\n"
            f"Recall: {self._recall:.4f}\n"
            f"F1 Score: {self._F1:.4f}\n"
            f"True Positives (TP): {self._TP}\n"
            f"True Negatives (TN): {self._TN}\n"
            f"False Positives (FP): {self._FP}\n"
            f"False Negatives (FN): {self._FN}"
        )
    
    def save(self, file_path: Optional[str] = "logs/metrics.csv", reset_logs: Optional[bool] = False):
        if not os.path.exists(file_path) or reset_logs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(f"Precision,Recall,F1,TP,TN,FP,FN\n")

        with open(file_path, "a") as f:
            f.write(
                f"{self._precision},{self._recall},{self._F1},{self._TP},{self._TN},{self._FP},{self._FN}\n"
            )


class ClassificationMetric(Metric):

    def __init__(self) -> None:
        super().__init__()

    def apply(self) -> None:
        self._precision = self._TP / (self._TP + self._FP + 1e-12)
        self._recall = self._TP / (self._TP + self._FN + 1e-12)
        self._F1 = 2 * (self._precision * self._recall) / (self._precision + self._recall + 1e-12)


class BinaryClassificationMetric(ClassificationMetric):

    __slots__ = [
        "_threshold",
    ]

    def __init__(self, threshold: Optional[float] = 0.5) -> None:
        super().__init__()
        self._threshold: float = threshold

    def step(self, output: torch.Tensor, labels: torch.Tensor) -> None:
        preds = output >= self._threshold
        labels = labels.bool()

        self._TP += ((preds == 1) & (labels == 1)).sum().item()
        self._FP += ((preds == 1) & (labels == 0)).sum().item()
        self._TN += ((preds == 0) & (labels == 0)).sum().item()
        self._FN += ((preds == 0) & (labels == 1)).sum().item()
