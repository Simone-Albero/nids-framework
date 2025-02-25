from typing import Dict, Any

import pandas as pd
import torch


def label_mapping(target: pd.Series) -> Dict[Any, int]:
    mapping = {label: idx for idx, label in enumerate(target.unique())}
    return mapping


def binary_label_mapping(target: pd.Series, target_label: Any) -> Dict[Any, int]:
    mapping = {label: 0 if label == target_label else 1 for label in target.unique()}
    return mapping


class OneHotEncoder:
    def __init__(self, levels: int):
        self.levels = levels

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=self.levels)

        if len(one_hot.shape) == 2:
            return one_hot.flatten()
        elif len(one_hot.shape) > 2:
            return one_hot.view(one_hot.size(0), -1)
        return one_hot
