from typing import Dict, Any, Tuple
import logging

import torch
import pandas as pd

from data_preparation.processor import DatasetProperties


def categorical_pre_processing(
    dataset: pd.DataFrame, properties: DatasetProperties, categorical_levels=32
) -> None:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags..."
    )

    for col in properties.categorical_features:
        unique_values = dataset[col].unique()[: (categorical_levels - 1)]
        value_map = {val: idx for idx, val in enumerate(unique_values)}

        dataset[col] = dataset[col].apply(lambda x: value_map.get(x, -1) + 1)


def bound_transformation(sample: Dict[str, Any], bound: int) -> Dict[str, Any]:
    features = sample["data"]
    features = torch.clamp(features, min=-bound, max=bound)

    logging.debug(f"Bound transformation result:\n{features}")

    sample["data"] = features
    return sample


def log_transformation(sample: Dict[str, Any], bound: int) -> Dict[str, Any]:
    features, stats = sample["data"], sample["stats"]

    min_values = torch.clamp(stats["min"], min=-bound)
    max_values = torch.clamp(stats["max"], max=bound)
    gaps = max_values - min_values

    mask = gaps != 0
    features = torch.where(
        mask, torch.log(features + 1) / torch.log(gaps + 1), torch.zeros_like(features)
    )
    features = torch.where(mask, features, torch.zeros_like(features))

    logging.debug(f"Log transformation result:\n{features}")

    sample["data"] = features
    return sample


# Deprecated
def categorical_value_encoding(
    sample: Dict[str, Any], categorical_bound: int
) -> Dict[str, Any]:
    features, stats = sample["data"], sample["stats"]
    categorical_levels = stats["categorical_levels"][:, :categorical_bound].t()

    value_encoding = torch.zeros_like(features)
    for col_idx, col in enumerate(features.t()):
        for row_idx, val in enumerate(col):
            mask = (categorical_levels[:, col_idx] == val).nonzero()
            if mask.numel() > 0:
                value_encoding[row_idx, col_idx] = mask.item() + 1

    sample["data"] = value_encoding
    return sample


def one_hot_encoding(sample: Dict[str, Any], categorical_levels: int) -> Dict[str, Any]:
    features = sample["data"]

    one_hot = torch.nn.functional.one_hot(features, num_classes=categorical_levels)
    logging.debug(f"One-hot transformation result:\n{one_hot}")

    if len(one_hot.shape) == 2:
        sample["data"] = one_hot.flatten()
    elif len(one_hot.shape) > 2:
        sample["data"] = one_hot.view(one_hot.size(0), -1)

    return sample
