import random
from typing import List, Iterator

import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset


class RandomSlidingWindowSampler(Sampler[List[int]]):

    __slots__ = [
        "window_size",
        "_dataset",
        "_indices",
        "_tot_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.window_size = window_size
        self._dataset = dataset
        self._indices: List[int] = list(range(len(dataset) - window_size + 1))
        self._tot_samples: int = len(self._indices)
        random.seed(42)

    def __iter__(self) -> Iterator[List[int]]:
        shuffled_indices = self._indices[:]
        random.shuffle(shuffled_indices)
        return iter(
            [
                torch.arange(start, start + self.window_size).tolist()
                for start in shuffled_indices
            ]
        )

    def __len__(self) -> int:
        return self._tot_samples


class IndexedSlidingWindowSampler(Sampler[List[int]]):

    __slots__ = [
        "window_size",
        "_dataset",
        "_indices",
        "_tot_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int, indices: pd.Series) -> None:
        self.window_size = window_size
        self._dataset = dataset
        self._indices = indices
        self._tot_samples: int = len(indices)

    def __iter__(self) -> Iterator[List[int]]:
        shuffled_indices = self._indices.sample(frac=1).reset_index(drop=True)

        return iter(
            [
                torch.arange(end - self.window_size + 1, end + 1).tolist()
                for end in shuffled_indices
            ]
        )

    def __len__(self) -> int:
        return self._tot_samples


class GroupWindowSampler(Sampler[List[int]]):

    __slots__ = [
        "window_size",
        "_dataset",
        "_indices",
        "_tot_samples",
    ]

    def __init__(
        self, dataset: Dataset, window_size: int, df: pd.DataFrame, group_column: str
    ) -> None:
        self._dataset: Dataset = dataset
        self.window_size: int = window_size
        random.seed(42)

        grouped_indices = {
            group: indices
            for group, indices in df.groupby(group_column).indices.items()
            if len(indices) >= window_size
        }

        self._indices: List[List[int]] = [
            indices[i : i + window_size]
            for indices in grouped_indices.values()
            for i in range(0, len(indices) - window_size + 1)
        ]

        self._tot_samples: int = len(self._indices)

    def __iter__(self) -> Iterator[List[int]]:
        shuffled_indices = self._indices[:]
        random.shuffle(shuffled_indices)
        for indices in shuffled_indices:
            yield indices

    def __len__(self) -> int:
        return self._tot_samples
