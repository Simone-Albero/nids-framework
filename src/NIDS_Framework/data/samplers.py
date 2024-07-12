from typing import Iterator, List, Any
import random

import pandas as pd
from torch.utils.data import Sampler, BatchSampler, Dataset
import torch


class RandomSlidingWindowSampler(Sampler):
    
    __slots__ = [
        "dataset",
        "window_size",
        "num_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.dataset = dataset
        self.window_size = window_size
        self.num_samples = len(dataset) - window_size + 1
        random.seed(42)

    def __iter__(self):
        indices = [
            random.randint(self.window_size - 1, len(self.dataset))
            for _ in range(self.num_samples)
        ]
        
        return iter([
            torch.arange(start - self.window_size, start)
            for start in indices
        ])

    def __len__(self):
        return self.num_samples

class FixedSlidingWindowSampler(Sampler):

    __slots__ = [
        "dataset",
        "window_size",
        "num_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.dataset = dataset
        self.window_size = window_size
        self.num_samples = len(dataset) // window_size
        random.seed(42)

    def __iter__(self):
        indices = [
            start_idx * self.window_size
            for start_idx in range(self.num_samples)
        ]
        random.shuffle(indices)
        
        return iter([
            torch.arange(start - self.window_size, start)
            for start in indices
        ])

    def __len__(self):
        return self.num_samples
    
class FairSlidingWindowSampler(Sampler):
    __slots__ = [
        "dataset",
        "window_size",
        "_malicious_indices",
        "_legit_indices"
        "num_samples",
    ]

    def __init__(self, dataset: Dataset, labels: pd.Series, benign_label: Any, window_size: int) -> None:
        if window_size % 2 != 0: raise ValueError("Window size must be an even number.")

        random.seed(42)
        self.dataset: Dataset = dataset
        self.window_size: int = window_size

        labels.reset_index(drop=True, inplace=True)
        malicious_mask = (labels != benign_label) & (labels.index > window_size - 1)
        legit_mask = (labels == benign_label) & (labels.index > window_size - 1)

        self._malicious_indices: List[int] = labels[malicious_mask].index.tolist()
        self._legit_indices: List[int] = labels[legit_mask].index.tolist()
        self.num_samples: int = 2 * min(len(self._malicious_indices), len(self._legit_indices))

    def __iter__(self) -> Iterator:
        random.shuffle(self._malicious_indices)
        random.shuffle(self._legit_indices)
        indices = []
        
        for malicious, legit in zip(self._malicious_indices, self._legit_indices):
            indices.append(list(range(malicious - self.window_size + 1, malicious + 1)))
            indices.append(list(range(legit - self.window_size + 1, legit + 1)))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
