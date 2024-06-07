from typing import List, Tuple, Optional, Dict
import logging

import pandas as pd
import torch

from data_preparation import tabular_modeling


class TabularDataset(tabular_modeling.TabularModeling):

    __slots__ = [
        "_stats",
    ]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__(numeric_data, categorical_data, labels)

        # max_len = max(len(categorical_data[col].value_counts().index) for col in categorical_data.columns)
        # categorical_levels = [
        #     list(categorical_data[col].value_counts().index) + [float('inf')] * (max_len - len(categorical_data[col].value_counts().index))
        #     for col in categorical_data.columns
        # ]

        self._stats: Dict[str, torch.Tensor] = {
            "mean": torch.tensor(numeric_data.mean().values, dtype=torch.float32),
            "std": torch.tensor(numeric_data.std().values, dtype=torch.float32),
            "min": torch.tensor(numeric_data.min().values, dtype=torch.float32),
            "max": torch.tensor(numeric_data.max().values, dtype=torch.float32),
            #'categorical_levels': torch.tensor(categorical_levels, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric_sample, categorical_sample, label_sample = self.applyTransformation(
            idx, self._stats
        )

        logging.debug(
            f"Numeric sample shape: {numeric_sample['data'].shape} Categorical sample shape: {categorical_sample['data'].shape}."
        )
        features = torch.cat(
            (numeric_sample["data"], categorical_sample["data"]), dim=-1
        )

        return features, label_sample["data"]
