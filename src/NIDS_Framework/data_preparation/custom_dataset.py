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

        max_len = max(len(categorical_data[col].value_counts().index) for col in categorical_data.columns)
        categorical_levels = [
            list(categorical_data[col].value_counts().index) + [float('inf')] * (max_len - len(categorical_data[col].value_counts().index))
            for col in categorical_data.columns
        ]

        self.stats = {
            'mean': torch.tensor(numeric_data.mean().values, dtype=torch.float32),
            'std': torch.tensor(numeric_data.std().values, dtype=torch.float32),
            'min': torch.tensor(numeric_data.min().values, dtype=torch.float32),
            'max': torch.tensor(numeric_data.max().values, dtype=torch.float32),
            'categorical_levels': torch.tensor(categorical_levels, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if self.labels is not None: label = torch.tensor(self.labels.iloc[idx].tolist(), dtype=torch.long)

        numeric_sample = {
                'features': torch.tensor(self.numeric_data.iloc[idx].values, dtype=torch.float32),
                'label': label,
                'stats': self.stats,
            }
        
        if self.numeric_transformation:
            numeric_sample = self.numeric_transformation(numeric_sample)

        categorical_sample = {
                'features': torch.tensor(self.categorical_data.iloc[idx].values, dtype=torch.float32),
                'label': label,
                'stats': self.stats,
            }
        if self.categorical_transformation:
            categorical_sample = self.categorical_transformation(categorical_sample)

        target_sample = {
                'label': label,
                'columns': self.categorical_data.columns,
                'stats': self.stats,
            }
        if self.target_transformation:
            target_sample = self.target_transformation(target_sample)

        print(numeric_sample['features'].shape, categorical_sample['features'].shape)

        return [numeric_sample['features'], categorical_sample['features']]
