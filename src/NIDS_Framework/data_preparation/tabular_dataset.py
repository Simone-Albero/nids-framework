from typing import List, Callable, Tuple, Optional
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TabularDataset(Dataset):

    __slots__ = [
        "numeric_data",
        "categorical_data",
        "labels",
        "_numeric_transformation",
        "_categorical_transformation",
        "_labels_transformation",
        "_stats",
    ]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
    ) -> None:
        self.numeric_data: torch.Tensor = torch.tensor(
            numeric_data.values, dtype=torch.float32
        )
        self.categorical_data: torch.Tensor = torch.tensor(
            categorical_data.values, dtype=torch.long
        )
        self.labels: torch.Tensor = torch.tensor(labels.values, dtype=torch.long)
        self._numeric_transformation: Compose = None
        self._categorical_transformation: Compose = None
        self._labels_transformation: Compose = None

        # max_len = max(len(categorical_data[col].value_counts().index) for col in categorical_data.columns)
        # categorical_levels = [
        #     list(categorical_data[col].value_counts().index) + [float('inf')] * (max_len - len(categorical_data[col].value_counts().index))
        #     for col in categorical_data.columns
        # ]

        self._stats = {
            "mean": torch.tensor(numeric_data.mean().values, dtype=torch.float32),
            "std": torch.tensor(numeric_data.std().values, dtype=torch.float32),
            "min": torch.tensor(numeric_data.min().values, dtype=torch.float32),
            "max": torch.tensor(numeric_data.max().values, dtype=torch.float32),
            #'categorical_levels': torch.tensor(categorical_levels, dtype=torch.float32),
        }

    @property
    def numeric_transformation(self) -> Compose | None:
        return self._numeric_transformation

    @numeric_transformation.setter
    def numeric_transformation(self, transformations: List[Callable]) -> None:
        self._numeric_transformation = Compose(transformations)

    @property
    def categorical_transformation(self) -> Compose | None:
        return self._categorical_transformation

    @categorical_transformation.setter
    def categorical_transformation(self, transformations: List[Callable]) -> None:
        self._categorical_transformation = Compose(transformations)

    @property
    def labels_transformation(self) -> Compose | None:
        return self._labels_transformation

    @labels_transformation.setter
    def labels_transformation(self, transformations: List[Callable]) -> None:
        self._labels_transformation = Compose(transformations)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric_sample = {
            "data": self.numeric_data[idx, :],
            "stats": self._stats,
        }

        if self._numeric_transformation:
            numeric_sample = self._numeric_transformation(numeric_sample)

        categorical_sample = {
            "data": self.categorical_data[idx, :],
            "stats": self._stats,
        }
        if self._categorical_transformation:
            categorical_sample = self._categorical_transformation(categorical_sample)

        label_sample = {
            "data": self.labels[idx],
            "stats": self._stats,
        }
        if self._labels_transformation:
            label_sample = self._labels_transformation(label_sample)

        logging.debug(
            f"Numeric sample shape: {numeric_sample['data'].shape} Categorical sample shape: {categorical_sample['data'].shape}."
        )
        features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        return features, label_sample["data"]
