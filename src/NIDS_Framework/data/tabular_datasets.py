from typing import List, Tuple, Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TabularDataset(Dataset):
    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        device: str,
        labels: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        self.numeric_data = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        self.categorical_data = torch.tensor(
            categorical_data.values, dtype=torch.long, device=device
        )

        if labels is not None:
            self.labels = torch.tensor(
                labels.values, dtype=torch.float32, device=device
            )
        else:
            self.labels = None

        self.stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

        self.numeric_transformation: Compose = []
        self.categorical_transformation: Compose = []
        self.labels_transformation: Compose = []

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric_sample = {"data": self.numeric_data[idx], "stats": self.stats}
        if self.numeric_transformation:
            numeric_sample = self.numeric_transformation(numeric_sample)

        categorical_sample = {"data": self.categorical_data[idx], "stats": self.stats}
        if self.categorical_transformation:
            categorical_sample = self.categorical_transformation(categorical_sample)

        label_sample = {"data": self.labels[idx], "stats": self.stats}
        if self.labels_transformation:
            label_sample = self.labels_transformation(label_sample)

        features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        return features, label_sample["data"][..., -1]

    def set_numeric_transformation(self, transformations: List[Callable]) -> None:
        self.numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: List[Callable]) -> None:
        self.categorical_transformation = Compose(transformations)

    def set_labels_transformation(self, transformations: List[Callable]) -> None:
        self.labels_transformation = Compose(transformations)
