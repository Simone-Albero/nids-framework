from typing import Iterator, List
import random

import pandas as pd
from torch.utils.data import Sampler, BatchSampler, Dataset


class RandomSlidingWindowSampler(Sampler):
    __slots__ = [
        "_dataset",
        "window_size",
        "_num_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self._dataset: Dataset = dataset
        self.window_size: int = window_size
        self._num_samples: int = len(dataset) - window_size + 1

    def __iter__(self) -> Iterator:
        start_indices = [
            random.randint(self.window_size - 1, len(self._dataset))
            for _ in range(len(self))
        ]

        return iter(
            [list(range(start - self.window_size, start)) for start in start_indices]
        )

    def __len__(self) -> int:
        return self._num_samples
    
class FairSlidingWindowSampler(Sampler):
    __slots__ = [
        "_dataset",
        "window_size",
        "_malicious_indices",
        "_legit_indices"
        "num_samples",
        "labels",
    ]

    def __init__(self, dataset: Dataset, labels: pd.Series, window_size: int) -> None:
        if window_size % 2 != 0: raise ValueError("Window size must be an even number.")

        random.seed(42)
        self.dataset: Dataset = dataset
        self.window_size: int = window_size

        labels.reset_index(drop=True, inplace=True)
        malicious_mask = (labels == 1) & (labels.index > window_size - 1)
        legit_mask = (labels == 0) & (labels.index > window_size - 1)

        self._malicious_indices: List[int] = labels[malicious_mask].index.tolist()
        self._legit_indices: List[int] = labels[legit_mask].index.tolist()
        self.labels = labels

        self.num_samples: int = min(len(self._malicious_indices), len(self._legit_indices))

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
    

# class BatchSampler(BatchSampler):
#     def __init__(self, sampler, batch_size, drop_last):
#         super().__init__(sampler, batch_size, drop_last)
        
#     def __iter__(self):
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
