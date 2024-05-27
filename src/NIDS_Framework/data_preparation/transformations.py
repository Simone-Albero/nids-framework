from typing import Any
import logging
from math import log

import torch


def bound_transformation(sample: Any, bound: int):
    """Convert numerical values to floats, and remove out of range values.

    Args:
        sample: The input sample.
        bound: The bound to filter values.
    """
    features = torch.tensor(sample["features"], dtype=torch.float32)
    features = torch.where(
        (features < -bound) | (features > bound), torch.tensor(0.0), features
    )

    logging.debug(f"Bound transformation: {features}")

    sample["features"] = features
    return sample


def log_transformation(sample: Any, bound: int):
    """Apply log transformation to the features in the sample.

    Args:
        sample: The input sample.
    """
    features, columns, stats = sample["features"], sample["columns"], sample["stats"]

    for i, column in enumerate(columns):
        min_val = max(stats["min"][column], -bound)
        max_val = min(stats["max"][column], bound)
        gap = max_val - min_val

        if gap == 0:
            features[i] = 0
        else:
            features[i] -= min_val
            features[i] = torch.log(features[i] + 1)
            features[i] *= 1.0 / log(gap + 1)

    logging.debug(f"Log transformation: {features}")
    return sample


def one_hot_transformation(sample: Any, categorical_bound: int):
    """One hot encode the features in the sample up to a given categorical bound.

    Args:
        categorical_bound: .
    """

    features, columns, stats = sample["features"], sample["columns"], sample["stats"]
    new_features = []

    for i, column in enumerate(columns):
        value_counts = stats["value_counts"][column]

        unique_values = value_counts.index.to_numpy()
        counts = value_counts.values

        sorted_unique_values = list(
            sorted(zip(unique_values, counts), key=lambda x: x[1], reverse=True)
        )
        top_unique_values = [x[0] for x in sorted_unique_values[:categorical_bound]]

        feature_encoding = torch.zeros(categorical_bound, dtype=torch.float32)
        if features[i] in top_unique_values:
            index = top_unique_values.index(features[i])
            feature_encoding[index] = 1.0

        new_features.append(feature_encoding)

    sample["features"] = new_features

    logging.debug(f'One Hot transformation: {sample["features"]}')
    return sample
