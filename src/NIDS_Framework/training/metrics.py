from typing import List
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class Metric(ABC):

    __slots__ = [
        "_preds",
        "_labels",
        "_precision",
        "_recall",
        "_accuracy",
        "_F1",
        "_TP",
        "_TN",
        "_FP",
        "_FN",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._preds: List[NDArray] = []
        self._labels: List[NDArray] = []

        self._precision: float = None
        self._recall: float = None
        self._accuracy: float = None
        self._F1: float = None
        self._TP: float = None
        self._TN: float = None
        self._FP: float = None
        self._FN: float = None

    def step(self, preds: List[NDArray], labels: List[NDArray]) -> None:
        self._preds.extend(preds)
        self._labels.extend(labels)

    @abstractmethod
    def apply(self) -> None:
        pass

    def __str__(self) -> str:
        return (
            f"Statistics:\n"
            f"Precision: {self._precision}\n"
            f"Recall: {self._recall}\n"
            f"Accuracy: {self._accuracy}\n"
            f"F1 Score: {self._F1}\n"
            f"True Positives (TP): {self._TP}\n"
            f"True Negatives (TN): {self._TN}\n"
            f"False Positives (FP): {self._FP}\n"
            f"False Negatives (FN): {self._FN}"
        )


class BinaryClassificationMetric(Metric):

    def __init__(self) -> None:
        super().__init__()

    def apply(self) -> None:
        self._accuracy = np.mean(recall_score(self._labels, self._preds, average=None))
        self._precision = precision_score(self._labels, self._preds, average="binary")
        self._recall = recall_score(self._labels, self._preds, average="binary")
        self._F1 = f1_score(self._labels, self._preds, average="binary")
        self._TN, self._FP, self._FN, self._TP = confusion_matrix(
            self._labels, self._preds
        ).ravel()
