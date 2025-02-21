import random
from typing import List, Iterator
import itertools

import pandas as pd
from torch.utils.data import Sampler, Dataset


class RandomSlidingWindowSampler(Sampler[List[int]]):
    __slots__ = ["window_size", "_dataset", "_indices"]

    def __init__(self, dataset: Dataset, window_size: int, seed: int = 42) -> None:
        self.window_size = window_size
        self._dataset = dataset
        self._indices: List[int] = list(range(len(dataset) - window_size + 1))
        random.seed(seed)

    def __iter__(self) -> Iterator[List[int]]:
        return (
            list(range(start, start + self.window_size))
            for start in random.sample(self._indices, len(self._indices))
        )

    def __len__(self) -> int:
        return len(self._indices)


class GroupWindowSampler(Sampler[List[int]]):
    __slots__ = ["window_size", "_dataset", "_indices"]

    def __init__(
        self, dataset, window_size: int, df: pd.DataFrame, group_column: str, seed: int = 42
    ) -> None:
        self._dataset = dataset
        self.window_size = window_size
        random.seed(seed)

        self._indices = list(
            itertools.chain.from_iterable(
                indices[i : i + window_size]
                for _, group in df.groupby(group_column)
                if len(group) >= window_size
                for indices in [group.index.tolist()]
                for i in range(len(indices) - window_size + 1)
            )
        )

    def __iter__(self) -> Iterator[List[int]]:
        random.shuffle(self._indices)
        yield from self._indices

    def __len__(self) -> int:
        return len(self._indices)
