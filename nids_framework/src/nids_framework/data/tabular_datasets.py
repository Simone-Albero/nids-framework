import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Callable, Dict, List, Optional, Tuple


class TabularDataset(Dataset):

    __slots__ = ["_data", "_target", "_transforms"]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        target: Optional[pd.DataFrame] = None,
        device: Optional[str] = "cpu",
        classification_type: str = "binary",
    ) -> None:
        if classification_type not in {"binary", "multiclass"}:
            raise ValueError("classification_type must be 'binary' or 'multiclass'")

        self._data: Dict[str, torch.Tensor] = {
            "numeric": torch.as_tensor(
                numeric_data.to_numpy(), dtype=torch.float32, device=device
            ),
            "categorical": torch.as_tensor(
                categorical_data.to_numpy(), dtype=torch.long, device=device
            ),
        }

        self._target: Optional[torch.Tensor] = None
        if target is not None:
            target_tensor = torch.as_tensor(target.to_numpy())
            self._target = target_tensor.squeeze().to(
                dtype=torch.float32 if classification_type == "binary" else torch.long,
                device=device,
            )

        self._transforms: Dict[str, Optional[Compose]] = {
            "numeric": None,
            "categorical": None,
            "target": None,
        }

    def __len__(self) -> int:
        return self._data["numeric"].shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        target_numeric = self._data["numeric"][idx]
        target_categorical = self._data["categorical"][idx]

        sample = {
            "numeric": self._data["numeric"][idx],
            "categorical": self._data["categorical"][idx],
            "target": self._target[idx] if self._target is not None else None,
        }

        for key in ["numeric", "categorical", "target"]:
            if self._transforms[key] and sample[key] is not None:
                sample[key] = self._transforms[key](sample[key])

        features = torch.cat((sample["numeric"], sample["categorical"]), dim=1)

        if self._target is not None:
            return features, sample["target"][..., -1]
        else:
            return features, target_numeric, target_categorical

    def set_transforms(self, transforms: Dict[str, List[Callable]]) -> None:
        for key in ["numeric", "categorical", "target"]:
            if key in transforms and transforms[key]:
                self._transforms[key] = Compose(transforms[key])
