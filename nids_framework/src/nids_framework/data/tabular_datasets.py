import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


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

        self._numeric_transformation: Compose = None
        self._categorical_transformation: Compose = None
        self._target_transformation: Compose = None

    def __len__(self) -> int:
        return len(self._target)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_sample = {"data": self._numeric_data[idx]}
        if self._numeric_transformation:
            numeric_sample = self._numeric_transformation(numeric_sample)

        categorical_sample = {"data": self._categorical_data[idx]}
        if self._categorical_transformation:
            categorical_sample = self._categorical_transformation(categorical_sample)

        target_sample = {"data": self._target[idx]}
        if self._target_transformation:
            target_sample = self._target_transformation(target_sample)

        categorical_sample["data"] = categorical_sample["data"].float()
        features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        return features, target_sample["data"][..., -1]

    def set_numeric_transformation(self, transformations: list[callable]) -> None:
        self._numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: list[callable]) -> None:
        self._categorical_transformation = Compose(transformations)

    def set_target_transformation(self, transformations: list[callable]) -> None:
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

        self._numeric_transformation: Compose = None
        self._categorical_transformation: Compose = None
        self._masking_transformation: Compose = None

    def __len__(self) -> int:
        return len(self._numeric_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_sample = {"data": self._numeric_data[idx]}
        if self._numeric_transformation:
            numeric_sample = self._numeric_transformation(numeric_sample)

        categorical_sample = {"data": self._categorical_data[idx]}
        if self._categorical_transformation:
            categorical_sample = self._categorical_transformation(categorical_sample)

        categorical_sample["data"] = categorical_sample["data"].float()
        originial_features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        original_sample = {"data": originial_features}
        if self._masking_transformation:
            masked_features = self._masking_transformation(original_sample)["data"]

        return masked_features, originial_features

    def set_numeric_transformation(self, transformations: list[callable]) -> None:
        self._numeric_transformation = Compose(transformations)

    def set_categorical_transformation(self, transformations: list[callable]) -> None:
        self._categorical_transformation = Compose(transformations)

    def set_masking_transformation(self, transformations: list[callable]) -> None:
        self._masking_transformation = Compose(transformations)

    def get_border(self) -> int:
        return self._numeric_data.shape[1]
