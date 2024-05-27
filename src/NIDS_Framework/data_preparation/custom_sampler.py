import random

from torch.utils.data import Sampler

     
class RandomSlidingWindowSampler(Sampler):
    __slots__ = [
        'dataset',
        'window_size',
        'num_samples',
    ]

    def __init__(self, dataset, window_size):
        self.dataset = dataset
        self.window_size = window_size
        self.num_samples = len(dataset) - window_size + 1

    def __iter__(self):
        start_indices = [random.randint(0, self.num_samples - 1) for _ in range(len(self))]
        return iter([list(range(start, start + self.window_size)) for start in start_indices])

    def __len__(self):
        return self.num_samples
