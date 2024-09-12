import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset


class RandomSlidingWindowSampler(Sampler):

    __slots__ = [
        "dataset",
        "window_size",
        "indices",
        "tot_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.dataset: Dataset = dataset
        self.window_size: int = window_size
        self.indices: List[int] = list(range(len(dataset) - window_size + 1))
        self.tot_samples: int = len(self.indices)
        random.seed(42)

    def __iter__(self):
        shuffled_indices = self.indices[:]
        random.shuffle(shuffled_indices)
        return iter(
            [
                torch.arange(start, start + self.window_size)
                for start in shuffled_indices
            ]
        )

    def __len__(self):
        return self.tot_samples


class IndexedSlidingWindowSampler(Sampler):

    __slots__ = [
        "dataset",
        "window_size",
        "indices",
        "tot_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int, indices: pd.Series) -> None:
        self.dataset: Dataset = dataset
        self.window_size: int = window_size
        self.indices: pd.Series = indices
        self.tot_samples: int = len(indices)

    def __iter__(self):
        shuffled_indices = self.indices.sample(frac=1).reset_index(drop=True)

        return iter(
            [
                torch.arange(end - self.window_size + 1, end + 1).tolist()
                for end in shuffled_indices
            ]
        )

    def __len__(self):
        return self.tot_samples


class GroupWindowSampler(Sampler):

    __slots__ = [
        "dataset",
        "window_size",
        "indices",
        "tot_samples",
    ]

    def __init__(
        self, dataset: Dataset, window_size: int, df: pd.DataFrame, group_column: str
    ) -> None:
        self.dataset: Dataset = dataset
        self.window_size: int = window_size
        random.seed(42)

        grouped_indices = {
            group: indices
            for group, indices in df.groupby(group_column).indices.items()
            if len(indices) >= window_size
        }

        self.indices: List[List[int]] = [
            indices[i : i + window_size]
            for indices in grouped_indices.values()
            for i in range(0, len(indices) - window_size + 1)
        ]

        self.tot_samples: int = len(self.indices)

    def __iter__(self):
        shuffled_indices = self.indices[:]
        random.shuffle(shuffled_indices)
        for indices in shuffled_indices:
            yield indices

    def __len__(self):
        return self.tot_samples
