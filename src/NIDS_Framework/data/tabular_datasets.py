from typing import List, Tuple, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class TabularDataset(Dataset):

    __slots__ = [
        "numeric_data",
        "categorical_data",
        "target",
        "numeric_transformation",
        "categorical_transformation",
        "target_transformation",
    ]
    
    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        target: pd.DataFrame,
        device: str = 'cpu',
        classification_type: str = 'binary'
    ) -> None:
        super().__init__()
        self.numeric_data = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        self.categorical_data = torch.tensor(
            categorical_data.values, dtype=torch.long, device=device
        )

        if classification_type == 'binary':
            self.target = torch.tensor(
                target.values.squeeze(), dtype=torch.float32, device=device
            )
        elif classification_type == 'multiclass':
            self.target = torch.tensor(
                target.values.squeeze(), dtype=torch.long, device=device
            )
        else:
            raise ValueError("classification_type must be 'binary' or 'multiclass'")

        self.numeric_transformation: Optional[Compose] = None
        self.categorical_transformation: Optional[Compose] = None
        self.target_transformation: Optional[Compose] = None

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric_sample = {"data": self.numeric_data[idx]}
        if self.numeric_transformation:
            numeric_sample = self.numeric_transformation(numeric_sample)

        categorical_sample = {"data": self.categorical_data[idx]}
        if self.categorical_transformation:
            categorical_sample = self.categorical_transformation(categorical_sample)

        target_sample = {"data": self.target[idx]}
        if self.target_transformation:
            target_sample = self.target_transformation(target_sample)

        categorical_sample["data"] = categorical_sample["data"].float()
        features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        return features, target_sample["data"][..., -1]

    def set_numeric_transformation(self, transformations: List[Callable]) -> None:
        self.numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: List[Callable]) -> None:
        self.categorical_transformation = Compose(transformations)

    def set_target_transformation(self, transformations: List[Callable]) -> None:
        self.target_transformation = Compose(transformations)