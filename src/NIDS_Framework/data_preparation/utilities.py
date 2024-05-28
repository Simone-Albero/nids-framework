from typing import Any
import logging

import torch


def bound_transformation(sample: Any, bound: int):
    """Convert numerical values to floats, and remove out of range values.

    Args:
        sample: The input sample.
        bound: The bound to filter values.
    """
    features = sample['features']
    features = torch.clamp(features,  min=-bound, max=bound)

    logging.debug(f'Bound transformation shape: {features.shape}')

    sample['features'] = features
    return sample


def log_transformation(sample: Any, bound: int):
    """Apply log transformation to the features in the sample.

    Args:
        sample: The input sample.
    """
    features, stats = sample['features'], sample['stats']
    
    min_values = torch.clamp(stats['min'],  min=-bound)
    max_values = torch.clamp(stats['max'],  max=bound)
    gaps = max_values - min_values

    mask = gaps != 0
    features = torch.where(mask, torch.log(features + 1) / torch.log(gaps + 1), torch.zeros_like(features))
    features = torch.where(mask, features, torch.zeros_like(features))
    
    logging.debug(f'Log transformation shape: {features.shape}')

    sample['features'] = features
    return sample


def categorical_value_encoding(sample: Any, categorical_bound: int):
    features, stats = sample['features'], sample['stats']
    categorical_levels = stats['categorical_levels'][:, :categorical_bound].t()

    value_encoding = torch.zeros_like(features)
    for col_idx, col in enumerate(features.t()):
        for row_idx, val in enumerate(col):
            mask = (categorical_levels[:, col_idx] == val).nonzero()
            if mask.numel() > 0:
                value_encoding[row_idx, col_idx] = mask.item() + 1

    logging.debug(f'Categorical value transformation shape: {value_encoding.shape}')

    sample['features'] = value_encoding
    return sample

def categorical_one_hot_encoding(sample: Any, categorical_bound: int):
    features = sample['features']

    one_hot_encode = lambda label: torch.eye(categorical_bound)[label.long()]
    one_hot_labels = torch.stack([one_hot_encode(label) for row in features for label in row])
    one_hot_labels = one_hot_labels.view(features.size(0), features.size(1), -1)


    logging.debug(f'Categorical one-hot transformation shape: {one_hot_labels.shape}')

    sample['features'] = one_hot_labels
    return sample

def collate_fn(batch):
    numeric = [item[0] for item in batch][0]
    categorical = [item[1] for item in batch][0].view(8, 288)
    labels = [item[2] for item in batch][0]

    print(numeric.shape, categorical.shape, labels.shape)

    return torch.cat((numeric, categorical), dim=1), labels