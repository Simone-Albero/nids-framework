import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Callable, List, Optional, Tuple


class TabularDataset(Dataset):

    __slots__ = [
        "_numeric_data",
        "_categorical_data",
        "_target",
        "_numeric_transformation",
        "_categorical_transformation",
        "_target_transformation",
    ]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        target: pd.DataFrame,
        device: str = "cpu",
        classification_type: str = "binary",
    ) -> None:
        self._numeric_data: torch.Tensor = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        self._categorical_data: torch.Tensor = torch.tensor(
            categorical_data.values, dtype=torch.long, device=device
        )

        if classification_type == "binary":
            self._target: torch.Tensor = torch.tensor(
                target.values.squeeze(), dtype=torch.float32, device=device
            )
        elif classification_type == "multiclass":
            self._target: torch.Tensor = torch.tensor(
                target.values.squeeze(), dtype=torch.long, device=device
            )
        else:
            raise ValueError("classification_type must be 'binary' or 'multiclass'")

        self._numeric_transformation: Optional[Compose] = None
        self._categorical_transformation: Optional[Compose] = None
        self._target_transformation: Optional[Compose] = None

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric = self._numeric_data[idx]
        if self._numeric_transformation:
            numeric = self._numeric_transformation(numeric)

        categorical = self._categorical_data[idx]
        if self._categorical_transformation:
            categorical = self._categorical_transformation(categorical)

        target = self._target[idx]
        if self._target_transformation:
            target = self._target_transformation(target)

        #categorical = categorical.float()
        features = torch.cat((numeric, categorical), dim=-1)

        return features, target[..., -1]

    def set_numeric_transformation(self, transformations: List[Callable]) -> None:
        self._numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: List[Callable]) -> None:
        self._categorical_transformation = Compose(transformations)

    def set_target_transformation(self, transformations: List[Callable]) -> None:
        self._target_transformation = Compose(transformations)


class TabularReconstructionDataset(Dataset):

    __slots__ = [
        "_numeric_data",
        "_categorical_data",
        "_numeric_transformation",
        "_categorical_transformation",
        "_masking_transformation",
    ]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        device: str = "cpu",
    ) -> None:
        self._numeric_data: torch.Tensor = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        self._categorical_data: torch.Tensor = torch.tensor(
            categorical_data.values, dtype=torch.long, device=device
        )

        self._numeric_transformation: Optional[Compose] = None
        self._categorical_transformation: Optional[Compose] = None
        self._masking_transformation: Optional[Compose] = None

    def __len__(self) -> int:
        return len(self._numeric_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric = self._numeric_data[idx]
        if self._numeric_transformation:
            numeric = self._numeric_transformation(numeric)

        categorical = self._categorical_data[idx]
        if self._categorical_transformation:
            categorical = self._categorical_transformation(categorical)

        #categorical = categorical.float()
        originial_features = torch.cat((numeric, categorical), dim=-1)

        masked_features = originial_features
        if self._masking_transformation:
            masked_features = self._masking_transformation(masked_features)

        return masked_features, originial_features

    def set_numeric_transformation(self, transformations: List[Callable]) -> None:
        self._numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: List[Callable]) -> None:
        self._categorical_transformation = Compose(transformations)

    def set_masking_transformation(self, transformations: List[Callable]) -> None:
        self._masking_transformation = Compose(transformations)

    def get_border(self) -> int:
        return self._numeric_data.shape[1]
