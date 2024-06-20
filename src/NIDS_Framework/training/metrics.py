from typing import Optional
from abc import ABC, abstractmethod

import torch


class Metric(ABC):

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
        self._precision: float = None
        self._recall: float = None
        self._F1: float = None
        self._TP: float = None
        self._TN: float = None
        self._FP: float = None
        self._FN: float = None

    @abstractmethod
    def step(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pass

    @abstractmethod
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


class ClassificationMetric(Metric):

    __slots__ = [
        "_preds",
        "_labels",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._preds: torch.Tensor = torch.empty(0).bool()
        self._labels: torch.Tensor = torch.empty(0).bool()

    def apply(self) -> None:
        self._TP = torch.count_nonzero(self._preds & self._labels).item()
        self._FP = torch.count_nonzero(self._preds & ~self._labels).item()
        self._TN = torch.count_nonzero(~self._preds & ~self._labels).item()
        self._FN = torch.count_nonzero(~self._preds & self._labels).item()

        self._precision = (
            self._TP / (self._TP + self._FP) if (self._TP + self._FP) > 0 else 0
        )
        self._recall = (
            self._TP / (self._TP + self._FN) if (self._TP + self._FN) > 0 else 0
        )
        self._F1 = (
            2 * (self._precision * self._recall) / (self._precision + self._recall)
            if (self._precision + self._recall) > 0
            else 0
        )


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

        self._preds = torch.cat((self._preds, preds), dim=0)
        self._labels = torch.cat((self._labels, labels), dim=0)


class MultipleClassificationMetric(ClassificationMetric):

    def __init__(self) -> None:
        super().__init__()

    def step(self, output: torch.Tensor, labels: torch.Tensor) -> None:
        output = torch.argmax(output, dim=1)

        self._preds = torch.cat((self._preds, output), dim=0)
        self._labels = torch.cat((self._labels, labels), dim=0)
