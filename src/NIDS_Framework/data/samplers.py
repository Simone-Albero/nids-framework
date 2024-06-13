from typing import Iterator
import random

from torch.utils.data import Sampler, Dataset


class RandomSlidingWindowSampler(Sampler):
    __slots__ = [
        "dataset",
        "window_size",
        "num_samples",
    ]

    def __init__(self, dataset: Dataset, window_size: int) -> None:
        self.dataset: Dataset = dataset
        self.window_size: int = window_size
        self.num_samples: int = len(dataset) - window_size + 1

    def __iter__(self) -> Iterator:
        start_indices = [
            random.randint(0, self.num_samples - 1) for _ in range(len(self))
        ]

        return iter(
            [list(range(start, start + self.window_size)) for start in start_indices]
        )

    def __len__(self) -> int:
        return self.num_samples
