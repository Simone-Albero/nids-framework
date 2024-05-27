from typing import Optional, Callable
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset implementation for handling pandas DataFrames.

    Attributes:
        numeric_data: The content of the Dataset as Pandas DataFrame.
        categorical_data: .
        labels: The labels of the Dataset as Pandas DataFrame.
        numeric_transformation: The Compose of numerical data transformations.
        categorical_transformation: The Compose of categorical data transformations.
        target_transformation: The Compose of labels transformations.
        stats: Dictionary to hold column-wise statistics for efficient transformation.
    """

    __slots__ = [
        'numeric_data',
        'categorical_data',
        'labels',
        'numeric_transformation',
        'categorical_transformation',
        'target_transformation',
        'stats',
    ]

    def __init__(
        self,
        numeric_data: pd.DataFrame,
        categorical_data: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
        numeric_transformation: Optional[Callable] = None,
        categorical_transformation: Optional[Callable] = None,
        target_transformation: Optional[Callable] = None,
    ) -> None:
        """Initialize the instance."""
        self.numeric_data = numeric_data
        self.categorical_data = categorical_data
        self.labels = labels
        self.numeric_transformation = numeric_transformation
        self.categorical_transformation = categorical_transformation
        self.target_transformation = target_transformation

        self.stats = {
            "mean": numeric_data.mean(),
            "std": numeric_data.std(),
            "min": numeric_data.min(),
            "max": numeric_data.max(),
            "value_counts": {
                col: categorical_data[col].value_counts()
                for col in categorical_data.columns
            },
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.labels is not None: label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        numeric_sample = {
                "features": self.numeric_data.iloc[idx].values,
                "label": label,
                "columns": self.numeric_data.columns,
                "stats": self.stats,
            }
        if self.numeric_transformation:
            numeric_sample = self.numeric_transformation(numeric_sample)

        categorical_sample = {
                "features": self.categorical_data.iloc[idx].values,
                "label": label,
                "columns": self.categorical_data.columns,
                "stats": self.stats,
            }
        if self.categorical_transformation:
            categorical_sample = self.categorical_transformation(categorical_sample)

        target_sample = {
                "label": label,
                "columns": self.categorical_data.columns,
                "stats": self.stats,
            }
        if self.target_transformation:
            target_sample = self.target_transformation(target_sample)

        numeric_tensor = numeric_sample["features"].unsqueeze(0)
        categorical_tensors = [
            categorical_feature.unsqueeze(0)
            for categorical_feature in categorical_sample["features"]
        ]
        features = torch.cat([numeric_tensor] + categorical_tensors, dim=1).squeeze(0)
        return features, label
